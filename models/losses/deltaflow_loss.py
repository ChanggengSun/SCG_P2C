import torch
import torch.nn as nn

from mmengine.registry import MODELS


def _safe_mean(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    return torch.nan_to_num(tensor.mean(), nan=0.0, posinf=0.0, neginf=0.0)


def _finite_pair_mask(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)


@MODELS.register_module()
class DeFlowLoss(nn.Module):
    """Original DeFlow supervised loss style."""

    def __init__(self):
        super().__init__()

    def forward(self, est_flow, gt_flow, gt_classes=None, gt_instance=None):
        mask = _finite_pair_mask(est_flow, gt_flow)
        pred = est_flow[mask]
        gt = gt_flow[mask]

        if pred.numel() == 0:
            return est_flow.sum() * 0.0

        pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)
        speed = torch.linalg.vector_norm(gt, dim=-1) / 0.1

        weight_loss = (
            _safe_mean(pts_loss[speed < 0.4])
            + _safe_mean(pts_loss[(speed >= 0.4) & (speed <= 1.0)])
            + _safe_mean(pts_loss[speed > 1.0])
        )
        return weight_loss


@MODELS.register_module()
class DeltaFlowLoss(nn.Module):
    """DeltaFlow-style class/instance-aware extension over DeFlow loss."""

    def __init__(self, class_weights=(0.1, 1.0, 2.0, 2.5, 1.5)):
        super().__init__()
        self.base_loss = DeFlowLoss()
        self.class_weights = class_weights

        # Category ids from dataprocess/common.py fallback mapping:
        # 0: NONE, 2: PEDESTRIAN, 3: STROLLER, 4: WHEELCHAIR,
        # 6: BICYCLE, 11: MOTORCYCLE, 7/8/9/10/12/13: vehicle-like
        self.vehicle_ids = torch.tensor([7, 8, 9, 10, 12, 13], dtype=torch.long)
        self.pedestrian_ids = torch.tensor([2, 3, 4], dtype=torch.long)
        self.wheeled_ids = torch.tensor([6, 11], dtype=torch.long)

    def _meta_category(self, classes: torch.Tensor) -> torch.Tensor:
        # 0: background, 1: car-like, 2: pedestrian, 3: wheeled, 4: other
        device = classes.device
        meta = torch.full_like(classes, fill_value=4, dtype=torch.long)
        meta[classes == 0] = 0
        meta[torch.isin(classes, self.vehicle_ids.to(device))] = 1
        meta[torch.isin(classes, self.pedestrian_ids.to(device))] = 2
        meta[torch.isin(classes, self.wheeled_ids.to(device))] = 3
        return meta

    def forward(self, est_flow, gt_flow, gt_classes=None, gt_instance=None):
        mask = _finite_pair_mask(est_flow, gt_flow)
        pred = est_flow[mask]
        gt = gt_flow[mask]
        if pred.numel() == 0:
            return est_flow.sum() * 0.0

        base = self.base_loss(pred, gt)

        if gt_classes is None:
            return base

        gt_classes = gt_classes[mask].long()
        pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)
        speed = torch.linalg.vector_norm(gt, dim=-1) / 0.1
        meta = self._meta_category(gt_classes)

        class_loss = pred.sum() * 0.0
        for class_id in range(5):
            class_mask = meta == class_id
            weighted_splits = (
                0.1 * _safe_mean(pts_loss[(speed < 0.4) & class_mask]),
                0.4 * _safe_mean(pts_loss[(speed >= 0.4) & (speed <= 1.0) & class_mask]),
                0.5 * _safe_mean(pts_loss[(speed > 1.0) & class_mask]),
            )
            class_loss = class_loss + sum(weighted_splits) * self.class_weights[class_id]

        instance_loss = pred.sum() * 0.0
        if gt_instance is not None:
            gt_instance = gt_instance[mask].long()
            valid_instance = gt_instance > 0
            unique_instances = torch.unique(gt_instance[valid_instance])
            cnt = 0
            for instance_id in unique_instances:
                ins_mask = gt_instance == instance_id
                if ins_mask.sum() == 0:
                    continue
                if speed[ins_mask].mean() <= 0.4:
                    continue
                class_id = int(torch.mode(meta[ins_mask], 0).values.item())
                ins_err = _safe_mean(pts_loss[ins_mask])
                instance_loss = instance_loss + ins_err * torch.exp(ins_err) * self.class_weights[class_id]
                cnt += 1
            if cnt > 0:
                instance_loss = instance_loss / cnt

        return base + class_loss + instance_loss
