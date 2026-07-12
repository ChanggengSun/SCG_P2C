import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS


class _ResidualConvExpert(nn.Module):
    def __init__(self, channels, dilation=1, depthwise=False):
        super().__init__()
        padding = dilation
        groups = channels if depthwise else 1
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.net(x)


class _GlobalContextExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = _ResidualConvExpert(channels)
        hidden = max(channels // 4, 16)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        local = self.local(x)
        return x + (local - x) * self.context(x)


@MODELS.register_module()
class BackboneTaskMoEAdapter(nn.Module):
    """Task-routed MoE adapter on top of shared backbone BEV features.

    The adapter keeps the backbone output shape unchanged. It routes the same
    BEV feature map through a shared expert pool with task-specific routers, so
    flow and tracking can receive different mixtures before their own fusers.
    """

    def __init__(
        self,
        in_channels=128,
        hidden_channels=128,
        num_experts=6,
        top_k=3,
        flow_prior=(0.05, 0.20, 0.35, 0.20, 0.05, 0.15),
        tracking_prior=(0.20, 0.25, 0.05, 0.20, 0.25, 0.05),
        balance_loss_weight=0.01,
        enable_pillar_motion_pred=True,
        dense_residual=0.05,
    ):
        super().__init__()
        if int(num_experts) != 6:
            raise ValueError('BackboneTaskMoEAdapter currently expects 6 experts.')
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.balance_loss_weight = float(balance_loss_weight)
        self.enable_pillar_motion_pred = bool(enable_pillar_motion_pred)
        self.dense_residual = max(0.0, min(float(dense_residual), 0.5))

        c = self.in_channels
        self.experts = nn.ModuleList([
            nn.Identity(),                         # 0 identity/shared
            _ResidualConvExpert(c),                # 1 foreground-like local filter
            _ResidualConvExpert(c),                # 2 local pillar motion filter
            _GlobalContextExpert(c),               # 3 global motion/context
            _ResidualConvExpert(c, depthwise=True), # 4 geometry/shape filter
            _ResidualConvExpert(c, dilation=2),     # 5 temporal-change-sensitive filter
        ])
        self.routers = nn.ModuleDict({
            'flow': nn.Conv2d(c, self.num_experts, kernel_size=1),
            'tracking': nn.Conv2d(c, self.num_experts, kernel_size=1),
        })

        self.pillar_motion_predictor = None
        if self.enable_pillar_motion_pred:
            self.pillar_motion_predictor = nn.Sequential(
                nn.Conv2d(c, self.hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_channels, 2, kernel_size=1),
            )

        self._init_router_bias('flow', flow_prior)
        self._init_router_bias('tracking', tracking_prior)

    def _init_router_bias(self, task, prior):
        prior = torch.as_tensor(prior, dtype=torch.float32)
        if prior.numel() != self.num_experts:
            raise ValueError(f'{task} prior must have {self.num_experts} values.')
        prior = prior.clamp_min(1e-4)
        prior = prior / prior.sum()
        with torch.no_grad():
            self.routers[task].bias.copy_(prior.log())
            self.routers[task].weight.zero_()

    def _normalize_task(self, task):
        task = str(task).lower()
        if task in ('track', 'tracking'):
            return 'tracking'
        if task == 'flow':
            return 'flow'
        raise ValueError(f'Unsupported MoE task: {task}')

    def _topk_gates(self, gates, return_stats=False):
        if self.top_k >= self.num_experts:
            if not return_stats:
                return gates
            selected_mask = torch.ones_like(gates)
            return gates, selected_mask
        top_vals, top_idx = torch.topk(gates, k=self.top_k, dim=1)
        selected_mask = torch.zeros_like(gates)
        selected_mask.scatter_(1, top_idx, 1.0)
        sparse = torch.zeros_like(gates)
        sparse.scatter_(1, top_idx, top_vals)
        sparse = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if self.dense_residual > 0:
            sparse = (1.0 - self.dense_residual) * sparse + self.dense_residual * gates
            sparse = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if return_stats:
            return sparse, selected_mask
        return sparse

    def _balance_loss(self, gates):
        importance = gates.mean(dim=(0, 2, 3))
        target = importance.new_full((self.num_experts,), 1.0 / self.num_experts)
        return ((importance - target) ** 2).sum() * self.balance_loss_weight

    def forward(self, x, task='flow', return_aux=False):
        task = self._normalize_task(task)
        logits = self.routers[task](x)
        dense_gates = F.softmax(logits, dim=1)
        gates, selected_mask = self._topk_gates(dense_gates, return_stats=True)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        routed = (expert_outputs * gates.unsqueeze(2)).sum(dim=1)

        if not return_aux:
            return routed

        raw_score_mean = dense_gates.detach().mean(dim=(0, 2, 3))
        final_weight_mean = gates.detach().mean(dim=(0, 2, 3))
        selection_rate = selected_mask.detach().mean(dim=(0, 2, 3))
        topk_scores, topk_ids = torch.topk(raw_score_mean, k=self.top_k, dim=0)
        aux = {
            'balance_loss': self._balance_loss(gates),
            'router_weights': gates.detach(),
            'expert_importance': final_weight_mean,
            'raw_score_mean': raw_score_mean,
            'final_weight_mean': final_weight_mean,
            'selection_rate': selection_rate,
            'raw_scores_map': dense_gates.detach(),
            'final_weights_map': gates.detach(),
            'selected_mask_map': selected_mask.detach(),
            'topk_ids': topk_ids.detach(),
            'topk_scores': topk_scores.detach(),
        }
        if task == 'flow' and self.pillar_motion_predictor is not None:
            # The pillar-motion auxiliary target explicitly supervises Expert 2.
            aux['pillar_motion_pred'] = self.pillar_motion_predictor(expert_outputs[:, 2])
        return routed, aux
