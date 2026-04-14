import numpy as np
import torch

from datasets import points_utils
from datasets.metrics import estimateOverlap, estimateAccuracy
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from nuscenes.utils import geometry_utils


@MODELS.register_module()
class P2PJointSeq5Voxel(BaseModel):
    """Joint model for synchronized Tracking+Flow with a shared backbone."""

    def __init__(
        self,
        backbone=None,
        fuser=None,
        tracking_head=None,
        flow_head=None,
        flow_loss=None,
        joint_pair_mode='seq5_pair123_flow_pair4',
        flow_freeze_shared_backbone=False,
        flow_backbone_lr_mult=1.0,
        cfg=None,
    ):
        super().__init__()
        self.config = cfg
        self.backbone = MODELS.build(backbone)
        self.fuse = MODELS.build(fuser)
        self.tracking_head = MODELS.build(tracking_head)
        self.flow_head = MODELS.build(flow_head)
        self.flow_loss = MODELS.build(flow_loss)
        self.joint_pair_mode = str(joint_pair_mode)
        self.flow_freeze_shared_backbone = bool(flow_freeze_shared_backbone)
        self.flow_backbone_lr_mult = float(flow_backbone_lr_mult)

        if self.joint_pair_mode == 'pc0_pc1':
            self._flow_pair_keys = [('pc0', 'pc1')]
            self._tracking_pair_index = 0
            self._two_frame_cycle = True
        else:
            self._flow_pair_keys = [('pch3', 'pch2'), ('pch2', 'pch1'), ('pch1', 'pc0'), ('pc0', 'pc1')]
            self._tracking_pair_index = 3
            self._two_frame_cycle = False

        head_num_pairs = int(getattr(self.flow_head, 'num_pairs', len(self._flow_pair_keys)))
        if head_num_pairs != len(self._flow_pair_keys):
            raise ValueError(
                f'flow_head.num_pairs ({head_num_pairs}) must match '
                f'pair mode requirements ({len(self._flow_pair_keys)}).')

        self._shared_lr_group_indices = None
        self._shared_lr_flow_base = None
        self._shared_lr_mixed_groups = False

    @staticmethod
    def _ensure_list(data):
        if isinstance(data, list):
            return data
        if isinstance(data, tuple):
            return list(data)
        if torch.is_tensor(data):
            if data.dim() >= 3:
                return [data[i] for i in range(data.shape[0])]
            return [data]
        return [data]

    @staticmethod
    def _merge_batch_list(batch_items):
        if len(batch_items) == 0:
            return {}

        first = batch_items[0]
        if isinstance(first, dict):
            merged = {}
            for key in first.keys():
                merged[key] = P2PJointSeq5Voxel._merge_batch_list([item[key] for item in batch_items])
            return merged

        return list(batch_items)

    @staticmethod
    def _ensure_sample_dict(data):
        if isinstance(data, (list, tuple)):
            data = list(data)
            if len(data) < 1:
                raise RuntimeError('Empty batch data.')
            if isinstance(data[0], dict):
                return P2PJointSeq5Voxel._merge_batch_list(data)
        if not isinstance(data, dict):
            raise TypeError(f'Invalid data type: {type(data)}')
        return data

    @staticmethod
    def _sum_loss_dict(loss_dict):
        vals = [v for k, v in loss_dict.items() if 'loss' in k and torch.is_tensor(v)]
        if len(vals) == 0:
            return next(iter(loss_dict.values())) * 0.0
        total = vals[0]
        for v in vals[1:]:
            total = total + v
        return total

    @staticmethod
    def _split_batch_items(data):
        if isinstance(data, list):
            return data
        if isinstance(data, tuple):
            return list(data)
        if torch.is_tensor(data):
            if data.dim() == 0:
                return [data]
            return [data[i] for i in range(data.shape[0])]
        return [data]

    @staticmethod
    def _as_pointcloud_list(data):
        # Point cloud tensors are per-sample (N, C); do not split along N.
        if isinstance(data, list):
            return data
        if isinstance(data, tuple):
            return list(data)
        return [data]

    def _iter_shared_parameters(self):
        for p in self.backbone.parameters():
            yield p
        for p in self.fuse.parameters():
            yield p

    def _freeze_shared_backbone(self):
        state = []
        for p in self._iter_shared_parameters():
            state.append((p, bool(p.requires_grad)))
            p.requires_grad = False
        return state

    @staticmethod
    def _restore_shared_backbone(state):
        for p, req in state:
            p.requires_grad = req

    def _init_shared_lr_groups(self, optim_wrapper):
        if self._shared_lr_group_indices is not None and self._shared_lr_flow_base is not None:
            return
        optimizer = getattr(optim_wrapper, 'optimizer', None)
        if optimizer is None:
            self._shared_lr_group_indices = []
            self._shared_lr_flow_base = {}
            return

        shared_ids = {id(p) for p in self._iter_shared_parameters()}
        group_indices = []
        flow_base = {}
        mixed = False

        for gi, group in enumerate(optimizer.param_groups):
            params = group.get('params', [])
            if params is None:
                continue
            param_ids = {id(p) for p in params}
            has_shared = len(param_ids & shared_ids) > 0
            if not has_shared:
                continue
            if len(param_ids - shared_ids) > 0:
                mixed = True
            group_indices.append(gi)
            flow_base[gi] = float(group.get('lr', 0.0))

        self._shared_lr_group_indices = group_indices
        self._shared_lr_flow_base = flow_base
        self._shared_lr_mixed_groups = mixed

    def _set_shared_lr_stage(self, optim_wrapper, stage):
        if self.flow_backbone_lr_mult <= 0:
            return
        self._init_shared_lr_groups(optim_wrapper)
        if self._shared_lr_mixed_groups:
            return

        optimizer = getattr(optim_wrapper, 'optimizer', None)
        if optimizer is None:
            return

        if stage == 'flow':
            scale = 1.0
        elif stage == 'tracking':
            scale = 1.0 / self.flow_backbone_lr_mult
        else:
            return

        for gi in self._shared_lr_group_indices:
            base_lr = self._shared_lr_flow_base.get(gi, None)
            if base_lr is None:
                continue
            optimizer.param_groups[gi]['lr'] = float(base_lr) * float(scale)

    def _normalize_track_field(self, data, num_pairs=4):
        outer = self._ensure_list(data)
        if len(outer) == 0:
            raise RuntimeError('Empty tracking field.')

        # sample-major: [batch][pair]
        if isinstance(outer[0], (list, tuple)):
            pair_major = [[] for _ in range(num_pairs)]
            for sample_item in outer:
                if not isinstance(sample_item, (list, tuple)) or len(sample_item) < num_pairs:
                    raise RuntimeError('Inconsistent sample-major tracking field.')
                for p in range(num_pairs):
                    pair_major[p].append(sample_item[p])
            return pair_major

        # pair-major: [pair][batch]
        if len(outer) >= num_pairs:
            pair_major = []
            for p in range(num_pairs):
                pair_major.append(self._split_batch_items(outer[p]))
            return pair_major

        raise RuntimeError(f'Cannot normalize tracking field with len={len(outer)}.')

    def _shared_pair_feats(self, prev_points, this_points):
        prev_points = self._as_pointcloud_list(prev_points)
        this_points = self._as_pointcloud_list(this_points)
        stack_points = prev_points + this_points
        stack_feats = self.backbone(stack_points)
        cat_feats = self.fuse(stack_feats)
        return cat_feats

    def _tracking_head_forward(self, feats, wlh):
        if self.config.box_aware:
            if isinstance(wlh, list):
                wlh = torch.stack(wlh)
            if wlh.dim() == 1:
                wlh = wlh.unsqueeze(0)

        # SyncBN workaround for batch=1
        if self.training and feats.shape[0] == 1:
            feats_bn = torch.cat([feats, feats], dim=0)
            if self.config.box_aware:
                wlh_bn = torch.cat([wlh, wlh], dim=0)
                res = self.tracking_head(feats_bn, wlh_bn)
            else:
                res = self.tracking_head(feats_bn)
            return {k: v[:1] for k, v in res.items()}

        if self.config.box_aware:
            return self.tracking_head(feats, wlh)
        return self.tracking_head(feats)

    def _build_tracking_context(self, inputs, data_samples):
        num_pairs = 1 if self._two_frame_cycle else 4
        prev_pair = self._normalize_track_field(inputs['track_prev_points'], num_pairs=num_pairs)
        this_pair = self._normalize_track_field(inputs['track_this_points'], num_pairs=num_pairs)
        wlh_pair = self._normalize_track_field(inputs['track_wlh'], num_pairs=num_pairs)
        box_pair = self._normalize_track_field(data_samples['track_box_label'], num_pairs=num_pairs)
        theta_pair = self._normalize_track_field(data_samples['track_theta'], num_pairs=num_pairs)

        context = []
        for i in range(num_pairs):
            bs = min(
                len(prev_pair[i]),
                len(this_pair[i]),
                len(wlh_pair[i]),
                len(box_pair[i]),
                len(theta_pair[i]),
            )
            if bs < 1:
                raise RuntimeError(f'Pair {i + 1} has empty batch.')
            context.append(
                dict(
                    prev_points=prev_pair[i][:bs],
                    this_points=this_pair[i][:bs],
                    wlh=wlh_pair[i][:bs],
                    box_label=box_pair[i][:bs],
                    theta=theta_pair[i][:bs],
                )
            )
        return context

    def _compute_tracking_loss_from_context(self, track_context, pair_indices=(0, 1, 2, 3)):
        batch_prev_points = []
        batch_this_points = []
        batch_wlh = []
        batch_box_label = []
        batch_theta = []
        sample_count = 0

        selected = sorted({int(i) for i in pair_indices if 0 <= int(i) < len(track_context)})
        if len(selected) < 1:
            fallback = track_context[0]['prev_points'][0]
            return fallback.sum() * 0.0, {}, 0

        for i in selected:
            pair_item = track_context[i]
            bs = min(
                len(pair_item['prev_points']),
                len(pair_item['this_points']),
                len(pair_item['wlh']),
                len(pair_item['box_label']),
                len(pair_item['theta']),
            )
            if bs < 1:
                continue

            batch_prev_points.extend(pair_item['prev_points'][:bs])
            batch_this_points.extend(pair_item['this_points'][:bs])
            batch_wlh.extend(pair_item['wlh'][:bs])
            batch_box_label.extend(pair_item['box_label'][:bs])
            batch_theta.extend(pair_item['theta'][:bs])
            sample_count += bs

        if sample_count < 1:
            fallback = track_context[0]['prev_points'][0]
            return fallback.sum() * 0.0, {}, 0

        feats = self._shared_pair_feats(batch_prev_points, batch_this_points)
        track_out = self._tracking_head_forward(feats, batch_wlh)
        loss_dict = self.tracking_head.loss(
            track_out,
            {
                'box_label': batch_box_label,
                'theta': batch_theta,
            },
        )
        total_loss = self._sum_loss_dict(loss_dict)
        return total_loss, loss_dict, sample_count

    def _compute_single_pair_tracking_loss(self, track_context, pair_index):
        pair_index = int(pair_index)
        if pair_index < 0 or pair_index >= len(track_context):
            raise IndexError(f'pair_index out of range: {pair_index}')

        pair_item = track_context[pair_index]
        bs = min(
            len(pair_item['prev_points']),
            len(pair_item['this_points']),
            len(pair_item['wlh']),
            len(pair_item['box_label']),
            len(pair_item['theta']),
        )
        if bs < 1:
            fallback = pair_item['prev_points'][0]
            return fallback.sum() * 0.0, {'regression_loss': fallback.sum() * 0.0}

        prev_points = pair_item['prev_points'][:bs]
        this_points = pair_item['this_points'][:bs]
        wlh = pair_item['wlh'][:bs]
        box_label = pair_item['box_label'][:bs]
        theta = pair_item['theta'][:bs]

        feats = self._shared_pair_feats(prev_points, this_points)
        track_out = self._tracking_head_forward(feats, wlh)
        loss_dict = self.tracking_head.loss(
            track_out,
            {
                'box_label': box_label,
                'theta': theta,
            },
        )
        total_loss = self._sum_loss_dict(loss_dict)
        return total_loss, loss_dict

    def _backward_step_once(self, loss, optim_wrapper):
        optim_wrapper.zero_grad()
        scale_loss = getattr(optim_wrapper, 'scale_loss', None)
        if callable(scale_loss):
            loss_scaled = scale_loss(loss)
        else:
            loss_scaled = loss
        optim_wrapper.backward(loss_scaled)
        optim_wrapper.step()
        optim_wrapper.zero_grad()

    def _compute_flow_pair_feats(self, inputs):
        return self._compute_flow_pair_feats_with_indices(inputs, None)

    def _compute_flow_pair_feats_with_indices(self, inputs, batch_indices=None):
        pair_feats = []

        for prev_key, this_key in self._flow_pair_keys:
            prev_list = self._ensure_list(inputs[prev_key])
            this_list = self._ensure_list(inputs[this_key])

            if batch_indices is not None:
                prev_list = [prev_list[i] for i in batch_indices if i < len(prev_list)]
                this_list = [this_list[i] for i in batch_indices if i < len(this_list)]

            bs = min(len(prev_list), len(this_list))
            if bs < 1:
                raise RuntimeError(f'No flow points for pair ({prev_key},{this_key})')
            # Batch-mode flow feature extraction:
            # run shared backbone/fuser once per temporal pair instead of once per sample.
            pair_feats.append(self._shared_pair_feats(prev_list[:bs], this_list[:bs]))

        return pair_feats

    def _compute_flow_loss(self, pair_feats, inputs, data_samples, batch_indices=None):
        query_points = inputs['query_points']
        if batch_indices is not None:
            query_list = self._ensure_list(query_points)
            query_points = [query_list[i] for i in batch_indices if i < len(query_list)]
        est_residual_list = self.flow_head(pair_feats, query_points)

        gt_flow_list = self._ensure_list(data_samples['gt_flow'])
        pose_flow_list = self._ensure_list(inputs['pose_flow'])
        valid_list = self._ensure_list(data_samples['flow_is_valid'])
        class_list = self._ensure_list(data_samples['flow_category_indices'])
        instance_list = self._ensure_list(data_samples['flow_instance_id'])
        if batch_indices is None:
            mapped_indices = list(range(len(est_residual_list)))
        else:
            mapped_indices = list(batch_indices)

        est_all, gt_all = [], []
        cls_all, ins_all = [], []

        for local_id, est_residual in enumerate(est_residual_list):
            src_id = mapped_indices[local_id] if local_id < len(mapped_indices) else local_id
            gt_flow = gt_flow_list[src_id].to(est_residual.device)
            pose_flow = pose_flow_list[src_id].to(est_residual.device)
            flow_valid = valid_list[src_id].to(est_residual.device).bool()
            flow_cls = class_list[src_id].to(est_residual.device).long()
            flow_ins = instance_list[src_id].to(est_residual.device).long()

            gt_residual = gt_flow - pose_flow
            finite_mask = torch.isfinite(est_residual).all(dim=-1) & torch.isfinite(gt_residual).all(dim=-1)
            train_mask = flow_valid & finite_mask
            if train_mask.sum() == 0:
                continue

            est_all.append(est_residual[train_mask])
            gt_all.append(gt_residual[train_mask])
            cls_all.append(flow_cls[train_mask])
            ins_all.append(flow_ins[train_mask])

        if len(est_all) == 0:
            return est_residual_list[0].sum() * 0.0

        est_all = torch.cat(est_all, dim=0)
        gt_all = torch.cat(gt_all, dim=0)
        cls_all = torch.cat(cls_all, dim=0) if len(cls_all) > 0 else None
        ins_all = torch.cat(ins_all, dim=0) if len(ins_all) > 0 else None

        return self.flow_loss(est_all, gt_all, cls_all, ins_all)

    def _get_flow_batch_indices(self, data_samples):
        flow_available = data_samples.get('flow_available', None)
        if flow_available is None:
            base = self._ensure_list(data_samples['gt_flow'])
            return list(range(len(base)))

        flags = self._ensure_list(flow_available)
        indices = []
        for i, flag in enumerate(flags):
            if torch.is_tensor(flag):
                is_true = bool(flag.detach().bool().item())
            else:
                is_true = bool(flag)
            if is_true:
                indices.append(i)
        return indices

    def forward(self, inputs, data_samples=None, mode='predict', task_type=None, **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        if mode == 'flow_predict':
            return self.flow_predict(inputs, data_samples)
        if mode == 'predict':
            return self.predict(inputs)
        raise RuntimeError(f'Invalid mode: {mode}')

    def loss(self, inputs, data_samples):
        # Loss-only path for compatibility/debug.
        if self._two_frame_cycle:
            track_context = self._build_tracking_context(inputs, data_samples)
            out = {}

            flow_indices = self._get_flow_batch_indices(data_samples)
            if len(flow_indices) > 0:
                flow_pair_feats = self._compute_flow_pair_feats_with_indices(inputs, flow_indices)
                flow_loss = self._compute_flow_loss(flow_pair_feats, inputs, data_samples, batch_indices=flow_indices)
            else:
                pair_total_fallback, _ = self._compute_single_pair_tracking_loss(track_context, self._tracking_pair_index)
                flow_loss = pair_total_fallback * 0.0
            out['flow_loss'] = flow_loss

            pair_total, pair_loss_dict = self._compute_single_pair_tracking_loss(track_context, self._tracking_pair_index)
            out['track_pair_loss'] = pair_total
            out[f'track_pair{self._tracking_pair_index + 1}_loss'] = pair_total
            for k, v in pair_loss_dict.items():
                out[f'track_pair_{k}'] = v
                out[f'track_pair{self._tracking_pair_index + 1}_{k}'] = v
            out['tracking_loss'] = pair_total
            return out

        # Legacy seq5 order:
        # pair1 -> pair2 -> pair3 -> flow -> pair4.
        track_context = self._build_tracking_context(inputs, data_samples)
        pair_losses = []
        out = {}
        for pair_idx in [0, 1, 2]:
            pair_total, pair_loss_dict = self._compute_single_pair_tracking_loss(track_context, pair_idx)
            pair_losses.append(pair_total)
            out[f'track_pair{pair_idx + 1}_loss'] = pair_total
            for k, v in pair_loss_dict.items():
                out[f'track_pair{pair_idx + 1}_{k}'] = v

        flow_indices = self._get_flow_batch_indices(data_samples)
        if len(flow_indices) > 0:
            flow_pair_feats = self._compute_flow_pair_feats_with_indices(inputs, flow_indices)
            flow_loss = self._compute_flow_loss(flow_pair_feats, inputs, data_samples, batch_indices=flow_indices)
        else:
            flow_loss = pair_losses[0] * 0.0
        out['flow_loss'] = flow_loss

        pair4_total, pair4_loss_dict = self._compute_single_pair_tracking_loss(track_context, 3)
        pair_losses.append(pair4_total)
        out['track_pair4_loss'] = pair4_total
        for k, v in pair4_loss_dict.items():
            out[f'track_pair4_{k}'] = v

        if len(pair_losses) > 0:
            tracking_loss = pair_losses[0]
            for v in pair_losses[1:]:
                tracking_loss = tracking_loss + v
            tracking_loss = tracking_loss / float(len(pair_losses))
        else:
            fallback_tensor = track_context[0]['prev_points'][0]
            tracking_loss = fallback_tensor.sum() * 0.0

        out['tracking_loss'] = tracking_loss
        return out

    def _predict_flow_outputs(self, pair_feats, inputs, data_samples):
        query_points = inputs['query_points']
        est_residual_list = self.flow_head(pair_feats, query_points)

        pc0_list = self._ensure_list(inputs['pc0'])
        gt_flow_list = self._ensure_list(data_samples['gt_flow'])
        pose_flow_list = self._ensure_list(inputs['pose_flow'])
        valid_list = self._ensure_list(data_samples['flow_is_valid'])
        class_list = self._ensure_list(data_samples['flow_category_indices'])

        bs = min(
            len(est_residual_list),
            len(pc0_list),
            len(gt_flow_list),
            len(pose_flow_list),
            len(valid_list),
            len(class_list),
        )
        outputs = []

        for batch_id in range(bs):
            est_residual = est_residual_list[batch_id]
            device = est_residual.device

            pc0 = pc0_list[batch_id].to(device)
            gt_flow = gt_flow_list[batch_id].to(device)
            pose_flow = pose_flow_list[batch_id].to(device)
            flow_valid = valid_list[batch_id].to(device).bool()
            flow_cls = class_list[batch_id].to(device).long()

            n = min(
                est_residual.shape[0],
                pc0.shape[0],
                gt_flow.shape[0],
                pose_flow.shape[0],
                flow_valid.shape[0],
                flow_cls.shape[0],
            )
            if n < 1:
                continue

            pred_flow = pose_flow[:n] + est_residual[:n]
            outputs.append(
                dict(
                    pred_flow=pred_flow.detach(),
                    rigid_flow=pose_flow[:n].detach(),
                    pc0=pc0[:n, :3].detach(),
                    gt_flow=gt_flow[:n].detach(),
                    flow_is_valid=flow_valid[:n].detach(),
                    flow_category_indices=flow_cls[:n].detach(),
                )
            )

        return outputs

    def flow_predict(self, inputs, data_samples=None):
        if data_samples is None:
            return []
        pair_feats = self._compute_flow_pair_feats(inputs)
        return self._predict_flow_outputs(pair_feats, inputs, data_samples)

    @staticmethod
    def _zero_from_inputs(inputs):
        preferred_keys = ['pc0', 'pc1', 'pch1', 'pch2', 'pch3']
        for key in preferred_keys:
            if key not in inputs:
                continue
            val = inputs[key]
            if isinstance(val, (list, tuple)):
                if len(val) < 1:
                    continue
                val = val[0]
            if torch.is_tensor(val):
                return val.sum() * 0.0
            try:
                t = torch.as_tensor(val, dtype=torch.float32)
                return t.sum() * 0.0
            except Exception:
                continue
        return torch.tensor(0.0, dtype=torch.float32)

    def train_step_flow(self, data, optim_wrapper):
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            data = self._ensure_sample_dict(data)
            inputs = data['inputs']
            data_samples = data['data_samples']

            self._set_shared_lr_stage(optim_wrapper, stage='flow')
            flow_indices = self._get_flow_batch_indices(data_samples)
            if len(flow_indices) > 0:
                flow_pair_feats = self._compute_flow_pair_feats_with_indices(inputs, flow_indices)
                flow_loss = self._compute_flow_loss(
                    flow_pair_feats, inputs, data_samples, batch_indices=flow_indices)
            else:
                flow_loss = self._zero_from_inputs(inputs)

        self._backward_step_once(flow_loss, optim_wrapper)

        flow_detached = flow_loss.detach()
        return {
            'flow_loss': flow_detached,
            'stage_updates': flow_detached.new_tensor(1.0),
        }

    def train_step_track(self, data, optim_wrapper):
        freeze_state = self._freeze_shared_backbone()
        try:
            with optim_wrapper.optim_context(self):
                data = self.data_preprocessor(data, True)
                data = self._ensure_sample_dict(data)
                inputs = data['inputs']
                data_samples = data['data_samples']

                self._set_shared_lr_stage(optim_wrapper, stage='tracking')
                track_context = self._build_tracking_context(inputs, data_samples)
                pair_total, pair_loss_dict = self._compute_single_pair_tracking_loss(
                    track_context, self._tracking_pair_index)

            self._backward_step_once(pair_total, optim_wrapper)
        finally:
            self._restore_shared_backbone(freeze_state)

        pair_detached = pair_total.detach()
        log_vars = {
            'track_pair_loss': pair_detached,
            f'track_pair{self._tracking_pair_index + 1}_loss': pair_detached,
            'tracking_loss': pair_detached,
            'stage_updates': pair_detached.new_tensor(1.0),
        }
        for k, v in pair_loss_dict.items():
            if torch.is_tensor(v):
                log_vars[f'track_pair_{k}'] = v.detach()
                log_vars[f'track_pair{self._tracking_pair_index + 1}_{k}'] = v.detach()
        return log_vars

    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data, True)
        data = self._ensure_sample_dict(data)
        inputs = data['inputs']
        data_samples = data['data_samples']

        if self._two_frame_cycle:
            log_vars = {}
            track_context = self._build_tracking_context(inputs, data_samples)

            # Stage 1: flow(pc0->pc1)
            flow_step_done = False
            flow_indices = self._get_flow_batch_indices(data_samples)
            if len(flow_indices) > 0:
                freeze_state = None
                if self.flow_freeze_shared_backbone:
                    freeze_state = self._freeze_shared_backbone()
                self._set_shared_lr_stage(optim_wrapper, stage='flow')
                try:
                    with optim_wrapper.optim_context(self):
                        flow_pair_feats = self._compute_flow_pair_feats_with_indices(inputs, flow_indices)
                        flow_loss = self._compute_flow_loss(
                            flow_pair_feats, inputs, data_samples, batch_indices=flow_indices)
                    self._backward_step_once(flow_loss, optim_wrapper)
                    flow_step_done = True
                finally:
                    if freeze_state is not None:
                        self._restore_shared_backbone(freeze_state)
            else:
                pair_total_fallback, _ = self._compute_single_pair_tracking_loss(track_context, self._tracking_pair_index)
                flow_loss = pair_total_fallback * 0.0
            log_vars['flow_loss'] = flow_loss.detach()

            # Stage 2: tracking(pc0->pc1)
            self._set_shared_lr_stage(optim_wrapper, stage='tracking')
            with optim_wrapper.optim_context(self):
                pair_total, pair_loss_dict = self._compute_single_pair_tracking_loss(
                    track_context, self._tracking_pair_index)
            self._backward_step_once(pair_total, optim_wrapper)
            tracking_loss = pair_total.detach()

            log_vars['track_pair_loss'] = tracking_loss
            log_vars[f'track_pair{self._tracking_pair_index + 1}_loss'] = tracking_loss
            for k, v in pair_loss_dict.items():
                if torch.is_tensor(v):
                    log_vars[f'track_pair_{k}'] = v.detach()
                    log_vars[f'track_pair{self._tracking_pair_index + 1}_{k}'] = v.detach()

            total_loss = tracking_loss + flow_loss.detach()
            log_vars['tracking_loss'] = tracking_loss
            log_vars['loss'] = total_loss
            log_vars['stage_updates'] = tracking_loss.new_tensor(2.0 if flow_step_done else 1.0)
            return log_vars

        # Legacy seq5 order:
        # pair1 -> pair2 -> pair3 -> flow -> pair4
        log_vars = {}
        pair_losses = []

        # Build unified tracking context once for the full seq5 sample.
        track_context = self._build_tracking_context(inputs, data_samples)

        for pair_idx in [0, 1, 2]:
            with optim_wrapper.optim_context(self):
                pair_total, pair_loss_dict = self._compute_single_pair_tracking_loss(track_context, pair_idx)
            self._backward_step_once(pair_total, optim_wrapper)

            pair_losses.append(pair_total.detach())
            log_vars[f'track_pair{pair_idx + 1}_loss'] = pair_total.detach()
            for k, v in pair_loss_dict.items():
                if torch.is_tensor(v):
                    log_vars[f'track_pair{pair_idx + 1}_{k}'] = v.detach()

        flow_step_done = False
        flow_indices = self._get_flow_batch_indices(data_samples)
        if len(flow_indices) > 0:
            with optim_wrapper.optim_context(self):
                flow_pair_feats = self._compute_flow_pair_feats_with_indices(inputs, flow_indices)
                flow_loss = self._compute_flow_loss(flow_pair_feats, inputs, data_samples, batch_indices=flow_indices)
            self._backward_step_once(flow_loss, optim_wrapper)
            flow_step_done = True
        else:
            flow_loss = pair_losses[0] * 0.0
        log_vars['flow_loss'] = flow_loss.detach()

        with optim_wrapper.optim_context(self):
            pair4_total, pair4_loss_dict = self._compute_single_pair_tracking_loss(track_context, 3)
        self._backward_step_once(pair4_total, optim_wrapper)
        pair_losses.append(pair4_total.detach())
        log_vars['track_pair4_loss'] = pair4_total.detach()
        for k, v in pair4_loss_dict.items():
            if torch.is_tensor(v):
                log_vars[f'track_pair4_{k}'] = v.detach()

        if len(pair_losses) > 0:
            tracking_loss = pair_losses[0]
            for v in pair_losses[1:]:
                tracking_loss = tracking_loss + v
            tracking_loss = tracking_loss / float(len(pair_losses))
        else:
            tracking_loss = flow_loss.detach() * 0.0

        total_loss = tracking_loss + flow_loss.detach()
        log_vars['tracking_loss'] = tracking_loss
        log_vars['loss'] = total_loss
        log_vars['stage_updates'] = flow_loss.detach().new_tensor(5.0 if flow_step_done else 4.0)
        return log_vars

    def _single_pair_inference(self, pair_input):
        feats = self._shared_pair_feats(pair_input['prev_points'][0], pair_input['this_points'][0])
        wlh = pair_input['wlh']
        out = self._tracking_head_forward(feats, wlh)
        coors = out['coors'][0]
        if self.config.use_rot:
            rot = out['rotation'][0]
            return coors, rot
        return coors

    def predict(self, inputs):
        # tracking-eval protocol (same as P2PVoxel / P2PJointVoxel)
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict) and '3d_bbox' in inputs[0]:
            ious = []
            distances = []
            results_bbs = []
            for frame_id in range(len(inputs)):
                this_bb = inputs[frame_id]['3d_bbox']

                if frame_id == 0:
                    results_bbs.append(this_bb)
                    last_coors = np.array([0.0, 0.0])
                else:
                    data_dict, ref_bb, flag = self.build_input_dict(inputs, frame_id, results_bbs)
                    if flag:
                        if self.config.use_rot:
                            coors, rot = self._single_pair_inference(data_dict)
                            rot = float(rot)
                        else:
                            coors = self._single_pair_inference(data_dict)
                            rot = 0.0
                        coors_x = float(coors[0])
                        coors_y = float(coors[1])
                        coors_z = float(coors[2])
                        last_coors = np.array([coors_x, coors_y])
                        candidate_box = points_utils.getOffsetBB(
                            ref_bb, [coors_x, coors_y, coors_z, rot],
                            degrees=True, use_z=True, limit_box=False)
                    else:
                        candidate_box = points_utils.getOffsetBB(
                            ref_bb, [last_coors[0], last_coors[1], 0, 0],
                            degrees=True, use_z=True, limit_box=False)
                    results_bbs.append(candidate_box)

                this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
                this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
                ious.append(this_overlap)
                distances.append(this_accuracy)

            return ious, distances

        return {'tracking': None, 'flow': None}

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0
        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]

        prev_pc = prev_frame['pc']
        this_pc = this_frame['pc']
        ref_box = results_bbs[-1]

        prev_frame_pc = points_utils.crop_pc_in_range(prev_pc, ref_box, self.config.point_cloud_range)
        this_frame_pc = points_utils.crop_pc_in_range(this_pc, ref_box, self.config.point_cloud_range)

        prev_points = prev_frame_pc.points.T
        this_points = this_frame_pc.points.T

        if self.config.post_processing is True:
            ref_bb = points_utils.transform_box(ref_box, ref_box)
            prev_idx = geometry_utils.points_in_box(ref_bb, prev_points.T, 1.25)
            if sum(prev_idx) < 3 and this_points.shape[0] < 25 and frame_id < 15:
                flag = False
            else:
                flag = True
        else:
            flag = True

        if prev_points.shape[0] < 1:
            if self.config.input_dim == 4:
                prev_points = np.zeros((1, 4), dtype='float32')
            elif self.config.input_dim == 5:
                prev_points = np.zeros((1, 5), dtype='float32')
            else:
                prev_points = np.zeros((1, 3), dtype='float32')

        if this_points.shape[0] < 1:
            if self.config.input_dim == 4:
                this_points = np.zeros((1, 4), dtype='float32')
            elif self.config.input_dim == 5:
                this_points = np.zeros((1, 5), dtype='float32')
            else:
                this_points = np.zeros((1, 3), dtype='float32')

        data_dict = {
            'prev_points': [torch.as_tensor(prev_points, dtype=torch.float32).cuda()],
            'this_points': [torch.as_tensor(this_points, dtype=torch.float32).cuda()],
            'wlh': torch.as_tensor(ref_box.wlh, dtype=torch.float32).cuda(),
        }
        return data_dict, results_bbs[-1], flag


