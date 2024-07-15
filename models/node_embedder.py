"""Neural network for embedding node features."""
import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding


class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s                        # final node embedding size
        self.c_pos_emb = self._cfg.c_pos_emb            # position embedding size
        self.c_timestep_emb = self._cfg.c_timestep_emb  # timestep embedding size
        self.linear = nn.Linear(
            self._cfg.c_pos_emb + self._cfg.c_timestep_emb, self.c_s)
        
        self.chain_idx_embed = nn.Linear(1, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, mask, pos, chain_id):
        # mask:     [batch, n_res] or [nres]
        # pos:      [batch, n_res] or [nres]
        # chain_id: [batch, n_res] or [nres]
        pos = pos.to(dtype=torch.float32).to(mask.device)
        chain_id = chain_id.to(dtype=torch.float32).to(mask.device)

        # [batch, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # [batch, n_res, c_timestep_emb]
        t_emb = self.embed_t(timesteps, mask)

        # [batch, n_res, c_s]
        pos_t_emb = self.linear(torch.cat([pos_emb, t_emb], dim=-1))

        # [batch, n_res, c_s]
        chain_id = chain_id.unsqueeze(-1)
        if chain_id.dim() == 2: # if original shape was [nres]
            chain_id = chain_id.unsqueeze(0)
        chain_idx_emb = self.chain_idx_embed(chain_id)

        # [batch, n_res, c_s]
        node_emb = pos_t_emb + chain_idx_emb
        
        return node_emb
