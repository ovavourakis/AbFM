"""Neural network for embedding node features."""
import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding


class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s                        # node embedding size
        self.c_pos_emb = self._cfg.c_pos_emb            # position embedding size
        self.c_timestep_emb = self._cfg.c_timestep_emb  # timestep embedding size
        self.linear = nn.Linear(
            self._cfg.c_pos_emb + self._cfg.c_timestep_emb, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, mask, pos):
        # mask: [batch, n_res]
        # pos: [batch, n_res]
        pos = pos.to(dtype=torch.float32).to(mask.device)
        # [batch, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * mask.unsqueeze(-1)

        input_feats = [pos_emb]
        input_feats.append(self.embed_t(timesteps, mask))

        return self.linear(torch.cat(input_feats, dim=-1))