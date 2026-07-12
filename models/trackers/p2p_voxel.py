import torch
from mmengine.model import BaseModel
from datasets.metrics import estimateOverlap, estimateAccuracy
import numpy as np
from datasets import points_utils
from nuscenes.utils import geometry_utils
from mmengine.registry import MODELS
from mmengine.logging import MMLogger


@MODELS.register_module()
class P2PVoxel(BaseModel):

    EXPERT_NAMES = (
        'Identity/Shared',
        'Foreground',
        'LocalPillarMotion',
        'GlobalMotion',
        'GeometryShape',
        'TemporalChange',
    )

    def __init__(self,
                 backbone=None,
                 fuser=None,
                 head=None,
                 backbone_moe=None,
                 moe_task='tracking',
                 moe_log_interval=50,
                 freeze_backbone=False,
                 cfg=None):
        super().__init__()
        self.config = cfg
        self.backbone = MODELS.build(backbone)
        self.fuse = MODELS.build(fuser)
        self.head = MODELS.build(head)
        self.backbone_moe = MODELS.build(backbone_moe) if backbone_moe is not None else None
        self.moe_task = str(moe_task)
        self.moe_log_interval = int(moe_log_interval)
        self._moe_call_count = 0
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self._freeze_backbone()
        return self

    def forward(self,
                inputs,
                data_samples=None,
                mode: str = 'predict',
                **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _tensor_to_list(self, value):
        if value is None:
            return []
        if torch.is_tensor(value):
            value = value.detach().float().cpu().tolist()
        return value

    def _format_expert_values(self, values):
        values = self._tensor_to_list(values)
        return ', '.join(
            f'E{i}:{name}={float(values[i]):.4f}'
            for i, name in enumerate(self.EXPERT_NAMES)
        )

    def _log_moe_aux(self, aux, input_shape=None, stage='train'):
        if not aux or self.moe_log_interval <= 0:
            return
        self._moe_call_count += 1
        if self._moe_call_count != 1 and self._moe_call_count % self.moe_log_interval != 0:
            return

        topk_ids = self._tensor_to_list(aux.get('topk_ids'))
        topk_scores = self._tensor_to_list(aux.get('topk_scores'))
        selected = []
        for expert_id, score in zip(topk_ids, topk_scores):
            expert_id = int(expert_id)
            selected.append(
                f'E{expert_id}:{self.EXPERT_NAMES[expert_id]}={float(score):.4f}'
            )

        msg = (
            f'[MoE][tracking][{stage}][call={self._moe_call_count}] '
            f'input_shape={input_shape} top_k={getattr(self.backbone_moe, "top_k", None)} | '
            f'all_raw_scores: {self._format_expert_values(aux.get("raw_score_mean"))} | '
            f'selected_topk_by_raw_score: [{", ".join(selected)}] | '
            f'selection_rate: {self._format_expert_values(aux.get("selection_rate"))} | '
            f'final_gate_mean: {self._format_expert_values(aux.get("final_weight_mean"))}'
        )
        try:
            MMLogger.get_current_instance().info(msg)
        except Exception:
            print(msg)

    def get_feats(self, inputs):
        prev_points = inputs['prev_points']
        this_points = inputs['this_points']
        stack_points = prev_points + this_points

        stack_feats = self.backbone(stack_points)
        moe_aux = None
        if self.backbone_moe is not None:
            if self.training:
                stack_feats, moe_aux = self.backbone_moe(
                    stack_feats, task=self.moe_task, return_aux=True)
                self._log_moe_aux(moe_aux, input_shape=tuple(stack_feats.shape), stage='train')
            else:
                stack_feats = self.backbone_moe(stack_feats, task=self.moe_task)

        cat_feats = self.fuse(stack_feats)
        if self.config.box_aware:
            wlh = torch.stack(inputs['wlh']) if isinstance(inputs['wlh'], list) \
                else inputs['wlh'].unsqueeze(0)
            results = self.head(cat_feats, wlh)
        else:
            results = self.head(cat_feats)

        if moe_aux is not None:
            results['moe_aux'] = moe_aux
        return results

    def inference(self, inputs):
        results = self.get_feats(inputs)
        coors = results['coors'][0]
        if self.config.use_rot:
            rot = results['rotation'][0]
            return coors, rot
        return coors

    def loss(self, inputs, data_samples):
        results = self.get_feats(inputs)
        losses = dict()
        losses.update(self.head.loss(results, data_samples))

        return losses

    def predict(self, inputs):
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(inputs)):  # tracklet
            this_bb = inputs[frame_id]["3d_bbox"]

            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
                last_coors = np.array([0., 0.])
            else:
                data_dict, ref_bb, flag = self.build_input_dict(inputs, frame_id, results_bbs)
                if flag:
                    if self.config.use_rot:
                        coors, rot = self.inference(data_dict)
                        rot = float(rot)
                    else:
                        coors = self.inference(data_dict)
                        rot = 0.
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

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

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
                # not enough points for tracking
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


        data_dict = {'prev_points': [torch.as_tensor(prev_points, dtype=torch.float32).cuda()],
                     'this_points': [torch.as_tensor(this_points, dtype=torch.float32).cuda()],
                     'wlh': torch.as_tensor(ref_box.wlh, dtype=torch.float32).cuda()
                     }

        return data_dict, results_bbs[-1], flag


