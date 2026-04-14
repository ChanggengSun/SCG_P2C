import torch
import torch.nn as nn

from mmengine.registry import MODELS


@MODELS.register_module()
class DeltaFlowTemporalHead(nn.Module):
    """Temporal flow head using shared pair features.

    Number of temporal pairs is controlled by ``num_pairs`` and must match
    the pair feature list provided by caller.
    """

    def __init__(self, in_channels=1024, hidden_channels=256, num_pairs=4):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_pairs = int(num_pairs)

        self.temporal_fuser = nn.Sequential(
            nn.Linear(self.in_channels * self.num_pairs, self.in_channels),
            nn.ReLU(True),
            nn.Linear(self.in_channels, self.in_channels),
            nn.ReLU(True),
        )

        self.point_mlp = nn.Sequential(
            nn.Linear(self.in_channels + 3, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, 3),
        )

    @staticmethod
    def _normalize_query_points(query_points):
        if isinstance(query_points, (list, tuple)):
            return list(query_points)
        if torch.is_tensor(query_points):
            if query_points.dim() == 2:
                return [query_points]
            if query_points.dim() == 3:
                return [query_points[i] for i in range(query_points.shape[0])]
            raise ValueError(f'Unsupported query_points shape: {query_points.shape}')
        raise TypeError(f'Unsupported query_points type: {type(query_points)}')

    def forward(self, pair_feats, query_points):
        if not isinstance(pair_feats, (list, tuple)) or len(pair_feats) != self.num_pairs:
            raise ValueError(f'pair_feats must be list/tuple with {self.num_pairs} elements.')

        query_points_list = self._normalize_query_points(query_points)

        pair_feats = [f if f.dim() == 2 else f.unsqueeze(0) for f in pair_feats]
        temporal_cat = torch.cat(pair_feats, dim=-1)
        temporal_ctx = self.temporal_fuser(temporal_cat)

        outputs = []
        for batch_id, points in enumerate(query_points_list):
            if points.dim() == 1:
                points = points.unsqueeze(0)
            points_xyz = points[:, :3]

            feat_id = min(batch_id, temporal_ctx.shape[0] - 1)
            repeated_ctx = temporal_ctx[feat_id].unsqueeze(0).expand(points_xyz.shape[0], -1)
            flow_pred = self.point_mlp(torch.cat([repeated_ctx, points_xyz], dim=1))
            outputs.append(flow_pred)
        return outputs
