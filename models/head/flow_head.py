import torch
import torch.nn as nn

from mmengine.registry import MODELS


@MODELS.register_module()
class FlowHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, 3),
        )

    def forward(self, global_feats, query_points):
        if isinstance(query_points, (list, tuple)):
            query_points_list = list(query_points)
        elif torch.is_tensor(query_points):
            if query_points.dim() == 2:
                query_points_list = [query_points]
            elif query_points.dim() == 3:
                query_points_list = [query_points[i] for i in range(query_points.shape[0])]
            else:
                raise ValueError(f'Unsupported query_points shape: {query_points.shape}')
        else:
            raise TypeError(f'Unsupported query_points type: {type(query_points)}')

        if global_feats.dim() == 1:
            global_feats = global_feats.unsqueeze(0)

        outputs = []
        for batch_id, points in enumerate(query_points_list):
            if points.dim() == 1:
                points = points.unsqueeze(0)
            points = points[:, :3]
            feat_id = min(batch_id, global_feats.shape[0] - 1)
            repeated_feat = global_feats[feat_id].unsqueeze(0).expand(points.shape[0], -1)
            flow_pred = self.mlp(torch.cat([repeated_feat, points], dim=1))
            outputs.append(flow_pred)
        return outputs
