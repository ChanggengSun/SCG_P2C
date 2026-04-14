import torch

from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module()
class P2PFlowSeq5Voxel(BaseModel):
    """Flow-only seq5 model with shared-structure backbone/fuser."""

    def __init__(
        self,
        backbone=None,
        fuser=None,
        flow_head=None,
        flow_loss=None,
        cfg=None,
    ):
        super().__init__()
        self.config = cfg
        self.backbone = MODELS.build(backbone)
        self.fuse = MODELS.build(fuser)
        self.flow_head = MODELS.build(flow_head)
        self.flow_loss = MODELS.build(flow_loss)

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
            keys = first.keys()
            for key in keys:
                merged[key] = P2PFlowSeq5Voxel._merge_batch_list([item[key] for item in batch_items])
            return merged

        return list(batch_items)

    @staticmethod
    def _ensure_sample_dict(data):
        if isinstance(data, (list, tuple)):
            data = list(data)
            if len(data) < 1:
                raise RuntimeError('Empty batch data.')
            if isinstance(data[0], dict):
                return P2PFlowSeq5Voxel._merge_batch_list(data)
        if not isinstance(data, dict):
            raise TypeError(f'Invalid data type: {type(data)}')
        return data

    def _shared_pair_feats(self, prev_points, this_points):
        stack_points = [prev_points, this_points]
        stack_feats = self.backbone(stack_points)
        cat_feats = self.fuse(stack_feats)
        return cat_feats

    def _compute_flow_pair_feats(self, inputs):
        pair_keys = [('pch3', 'pch2'), ('pch2', 'pch1'), ('pch1', 'pc0'), ('pc0', 'pc1')]
        pair_feats = []
        for prev_key, this_key in pair_keys:
            prev_list = self._ensure_list(inputs[prev_key])
            this_list = self._ensure_list(inputs[this_key])
            bs = min(len(prev_list), len(this_list))
            if bs < 1:
                raise RuntimeError(f'No flow points for pair ({prev_key},{this_key})')

            feats_b = []
            for b in range(bs):
                feats_b.append(self._shared_pair_feats(prev_list[b], this_list[b]))
            if len(feats_b) == 1:
                pair_feats.append(feats_b[0])
            else:
                pair_feats.append(torch.cat(feats_b, dim=0))
        return pair_feats

    def _compute_flow_loss(self, pair_feats, inputs, data_samples):
        query_points = inputs['query_points']
        est_residual_list = self.flow_head(pair_feats, query_points)

        gt_flow_list = self._ensure_list(data_samples['gt_flow'])
        pose_flow_list = self._ensure_list(inputs['pose_flow'])
        valid_list = self._ensure_list(data_samples['flow_is_valid'])
        class_list = self._ensure_list(data_samples['flow_category_indices'])
        instance_list = self._ensure_list(data_samples['flow_instance_id'])

        est_all, gt_all = [], []
        cls_all, ins_all = [], []

        for batch_id, est_residual in enumerate(est_residual_list):
            gt_flow = gt_flow_list[batch_id].to(est_residual.device)
            pose_flow = pose_flow_list[batch_id].to(est_residual.device)
            flow_valid = valid_list[batch_id].to(est_residual.device).bool()
            flow_cls = class_list[batch_id].to(est_residual.device).long()
            flow_ins = instance_list[batch_id].to(est_residual.device).long()

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

    def forward(self, inputs, data_samples=None, mode='predict', **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        raise RuntimeError(f'Invalid mode: {mode}')

    def loss(self, inputs, data_samples):
        pair_feats = self._compute_flow_pair_feats(inputs)
        flow_loss = self._compute_flow_loss(pair_feats, inputs, data_samples)
        return {'flow_loss': flow_loss}

    def predict(self, inputs, data_samples=None):
        if data_samples is None:
            return []
        pair_feats = self._compute_flow_pair_feats(inputs)
        return self._predict_flow_outputs(pair_feats, inputs, data_samples)

    def train_step(self, data, optim_wrapper):
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            data = self._ensure_sample_dict(data)
            losses = self(**data, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data):
        data = self.data_preprocessor(data, False)
        data = self._ensure_sample_dict(data)
        return self(**data, mode='predict')

    def test_step(self, data):
        data = self.data_preprocessor(data, False)
        data = self._ensure_sample_dict(data)
        return self(**data, mode='predict')
