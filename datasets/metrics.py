import numpy as np
import os
from pathlib import Path
import sys
import types
import torch
import torchmetrics.utilities.data
from shapely.geometry import Polygon
from torchmetrics import Metric
from mmengine.evaluator import BaseMetric
import torchmetrics.utilities.data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def estimateAccuracy(box_a, box_b, dim=3, up_axis=(0, -1, 0)):
    if dim == 3:
        return np.linalg.norm(box_a.center - box_b.center, ord=2)
    elif dim == 2:
        up_axis = np.array(up_axis)
        return np.linalg.norm(
            box_a.center[up_axis != 0] - box_b.center[up_axis != 0], ord=2)


def fromBoxToPoly(box, up_axis=(0, -1, 0)):
    """

    :param box:
    :param up_axis: the up axis must contain only one non-zero component
    :return:
    """
    if up_axis[1] != 0:
        return Polygon(tuple(box.corners()[[0, 2]].T[[0, 1, 5, 4]]))
    elif up_axis[2] != 0:
        return Polygon(tuple(box.bottom_corners().T))


def estimateOverlap(box_a, box_b, dim=2, up_axis=(0, -1, 0)):
    # if box_a == box_b:
    #     return 1.0
    try:
        Poly_anno = fromBoxToPoly(box_a, up_axis)
        Poly_subm = fromBoxToPoly(box_b, up_axis)

        box_inter = Poly_anno.intersection(Poly_subm)
        box_union = Poly_anno.union(Poly_subm)
        if dim == 2:
            return box_inter.area / box_union.area

        else:
            up_axis = np.array(up_axis)
            up_max = min(box_a.center[up_axis != 0], box_b.center[up_axis != 0])
            up_min = max(box_a.center[up_axis != 0] - box_a.wlh[2], box_b.center[up_axis != 0] - box_b.wlh[2])
            inter_vol = box_inter.area * max(0, up_max[0] - up_min[0])
            anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
            subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]

            overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
            return overlap
    except ValueError:
        return 0.0


class TorchPrecision(Metric):
    """Computes and stores the Precision using torchMetrics"""

    def __init__(self, n=21, max_accuracy=2, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_accuracy = max_accuracy
        self.Xaxis = torch.linspace(0, self.max_accuracy, steps=n)
        self.add_state("accuracies", default=[])

    def value(self, accs):
        prec = [
            torch.sum((accs <= thres).float()) / len(accs)
            for thres in self.Xaxis
        ]
        return torch.tensor(prec)

    def update(self, val):
        self.accuracies.append(val)

    def compute(self):
        accs = torchmetrics.utilities.data.dim_zero_cat(self.accuracies)
        if accs.numel() == 0:
            return 0
        return torch.trapz(self.value(accs), x=self.Xaxis) * 100 / self.max_accuracy


class TorchSuccess(Metric):
    """Computes and stores the Success using torchMetrics"""

    def __init__(self, n=21, max_overlap=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_overlap = max_overlap
        self.Xaxis = torch.linspace(0, self.max_overlap, steps=n)
        self.add_state("overlaps", default=[])

    def value(self, overlaps):
        succ = [
            torch.sum((overlaps >= thres).float()) / len(overlaps)
            for thres in self.Xaxis
        ]
        return torch.tensor(succ)

    def compute(self):
        overlaps = torchmetrics.utilities.data.dim_zero_cat(self.overlaps)

        if overlaps.numel() == 0:
            return 0
        return torch.trapz(self.value(overlaps), x=self.Xaxis) * 100 / self.max_overlap

    def update(self, val):
        self.overlaps.append(val)


class TrackAccuracy(BaseMetric):

    def __init__(self,
                 n=21,
                 max_accuracy=2,
                 max_overlap=1,
                 **kargs):
        super().__init__(**kargs)
        self.max_accuracy = max_accuracy
        self.max_overlap = max_overlap
        self.prec_axis = torch.linspace(0, self.max_accuracy, steps=n)
        self.succ_axis = torch.linspace(0, self.max_overlap, steps=n)

    def process(self, data_batch, data_samples):
        overlap, distance = torch.tensor(data_samples)
        self.results.append((distance, overlap))

    def compute_metrics(self, results):
        distance, overlap = zip(*results)

        distance = torchmetrics.utilities.data.dim_zero_cat(distance)
        overlap = torchmetrics.utilities.data.dim_zero_cat(overlap)

        prec = torch.tensor([torch.sum(
            (distance <= thres).float()) / len(distance) for thres in self.prec_axis])
        succ = torch.tensor([torch.sum(
            (overlap >= thres).float()) / len(overlap) for thres in self.succ_axis])

        success = torch.trapz(succ, x=self.succ_axis) * 100 / self.max_overlap if overlap.numel() > 0 else 0
        precision = torch.trapz(prec, x=self.prec_axis) * 100 / self.max_accuracy if distance.numel() > 0 else 0

        return dict(
            success=success,
            precision=precision
        )


class FlowEPEMetric(BaseMetric):
    """Flow validation metric on valid points only."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples):
        if isinstance(data_samples, dict):
            outputs = [data_samples]
        elif isinstance(data_samples, (list, tuple)):
            outputs = list(data_samples)
        else:
            outputs = []

        for out in outputs:
            if not isinstance(out, dict):
                continue
            if 'flow_epe_sum' not in out or 'flow_epe_count' not in out:
                continue
            flow_sum = float(torch.as_tensor(out['flow_epe_sum']).detach().cpu().item())
            flow_count = float(torch.as_tensor(out['flow_epe_count']).detach().cpu().item())
            self.results.append((flow_sum, flow_count))

    def compute_metrics(self, results):
        total_sum = 0.0
        total_count = 0.0
        for flow_sum, flow_count in results:
            total_sum += float(flow_sum)
            total_count += float(flow_count)

        flow_epe = total_sum / max(total_count, 1.0)
        return dict(flow_epe=flow_epe)


def _load_single_task_flow_eval(flow_repo_root):
    root = Path(flow_repo_root)
    if not root.exists():
        raise FileNotFoundError(f'OpenSceneFlow root does not exist: {root}')

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # `tabulate` is only used for pretty printing in single-task eval code.
    if 'tabulate' not in sys.modules:
        tabulate_stub = types.ModuleType('tabulate')
        tabulate_stub.tabulate = lambda *args, **kwargs: ''
        sys.modules['tabulate'] = tabulate_stub

    try:
        from src.utils.eval_metric import (
            OfficialMetrics,
            evaluate_leaderboard,
            evaluate_leaderboard_v2,
            evaluate_ssf,
        )
    except Exception as e:
        raise RuntimeError(
            'Failed to import official flow evaluation from single-task OpenSceneFlow. '
            f'root={root_str}. Original error: {e}'
        ) from e

    return OfficialMetrics, evaluate_leaderboard, evaluate_leaderboard_v2, evaluate_ssf


class FlowOfficialMetric(BaseMetric):
    """Use the same flow validation metrics as single-task OpenSceneFlow."""

    def __init__(self, flow_repo_root=None, log_ssf=False, **kwargs):
        super().__init__(**kwargs)
        if flow_repo_root is None:
            flow_repo_root = os.environ.get('OPENSCENEFLOW_ROOT', 'C:/develop/OpenSceneFlow')
        self.flow_repo_root = flow_repo_root
        self.log_ssf = bool(log_ssf)

        (self._OfficialMetrics,
         self._evaluate_leaderboard,
         self._evaluate_leaderboard_v2,
         self._evaluate_ssf) = _load_single_task_flow_eval(self.flow_repo_root)

    @staticmethod
    def _to_outputs(data_samples):
        if isinstance(data_samples, dict):
            return [data_samples]
        if isinstance(data_samples, (list, tuple)):
            return [x for x in data_samples if isinstance(x, dict)]
        return []

    @staticmethod
    def _to_tensor(value):
        return torch.as_tensor(value).detach().cpu()

    def process(self, data_batch, data_samples):
        outputs = self._to_outputs(data_samples)
        for out in outputs:
            required = [
                'pred_flow',
                'rigid_flow',
                'pc0',
                'gt_flow',
                'flow_is_valid',
                'flow_category_indices',
            ]
            if any(k not in out for k in required):
                continue

            pred_flow = self._to_tensor(out['pred_flow'])
            rigid_flow = self._to_tensor(out['rigid_flow'])
            pc0 = self._to_tensor(out['pc0'])
            gt_flow = self._to_tensor(out['gt_flow'])
            flow_is_valid = self._to_tensor(out['flow_is_valid'])
            flow_category_indices = self._to_tensor(out['flow_category_indices'])

            if pred_flow.ndim != 2 or gt_flow.ndim != 2 or pc0.ndim != 2:
                continue

            n = min(
                pred_flow.shape[0],
                rigid_flow.shape[0],
                pc0.shape[0],
                gt_flow.shape[0],
                flow_is_valid.shape[0],
                flow_category_indices.shape[0],
            )
            if n < 1:
                continue

            pred_flow = pred_flow[:n, :3].float()
            rigid_flow = rigid_flow[:n, :3].float()
            pc0 = pc0[:n, :3].float()
            gt_flow = gt_flow[:n, :3].float()
            flow_is_valid = flow_is_valid[:n].bool()
            flow_category_indices = flow_category_indices[:n].long()

            if int(flow_is_valid.sum().item()) < 1:
                continue

            try:
                v1_dict = self._evaluate_leaderboard(
                    pred_flow,
                    rigid_flow,
                    pc0,
                    gt_flow,
                    flow_is_valid,
                    flow_category_indices,
                )
                v2_dict = self._evaluate_leaderboard_v2(
                    pred_flow,
                    rigid_flow,
                    pc0,
                    gt_flow,
                    flow_is_valid,
                    flow_category_indices,
                )
                ssf_dict = self._evaluate_ssf(
                    pred_flow,
                    rigid_flow,
                    pc0,
                    gt_flow,
                    flow_is_valid,
                    flow_category_indices,
                )
            except Exception:
                continue

            self.results.append((v1_dict, v2_dict, ssf_dict))

    def compute_metrics(self, results):
        if len(results) == 0:
            return {'val/Dynamic/Mean': float('inf')}

        metrics = self._OfficialMetrics()
        for v1_dict, v2_dict, ssf_dict in results:
            metrics.step(v1_dict, v2_dict, ssf_dict)

        metrics.normalize()

        out = {}
        for key in metrics.bucketed:
            out[f'val/Static/{key}'] = float(metrics.bucketed[key]['Static'])
            out[f'val/Dynamic/{key}'] = float(metrics.bucketed[key]['Dynamic'])

        for key, value in metrics.epe_3way.items():
            out[f'val/{key}'] = float(value)

        if self.log_ssf:
            for dis_key, values in metrics.epe_ssf.items():
                out[f'val/SSF/{dis_key}/Static'] = float(values['Static'])
                out[f'val/SSF/{dis_key}/Dynamic'] = float(values['Dynamic'])

        dynamic_mean = out.get('val/Dynamic/Mean', float('inf'))
        if not np.isfinite(dynamic_mean):
            out['val/Dynamic/Mean'] = float('inf')

        return out
