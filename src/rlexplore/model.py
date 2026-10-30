import math
import torch
from torch import nn
from torch.distributions import Categorical
from .env import STATE_DIM, PRIV_DIM, VIEWS


class FastWeightMemory(nn.Module):
    """Small TTT-linear-style fast state, NOT a reproduction of the TTT paper.

    Each token uses a learned key/value self-supervised regression update.
    Fast weights are episode-local and recomputed from history on every forward;
    outer-loop gradients pass through this differentiable update during PPO.
    """
    def __init__(self, dim):
        super().__init__()
        self.key, self.query, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.rate = nn.Parameter(torch.tensor(-3.))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, valid):
        b, t, d = x.shape
        weights = x.new_zeros(b, d, d)
        outputs = []
        for i in range(t):
            k = torch.nn.functional.normalize(self.key(x[:, i]), dim=-1)
            q = torch.nn.functional.normalize(self.query(x[:, i]), dim=-1)
            v = self.value(x[:, i])
            prediction = torch.einsum("bde,be->bd", weights, k)
            update = (v-prediction).unsqueeze(-1)*k.unsqueeze(1)
            weights = weights + torch.sigmoid(self.rate)*update*valid[:, i, None, None]
            outputs.append(self.norm(x[:, i]+torch.einsum("bde,be->bd", weights, q)))
        return torch.stack(outputs, 1)


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.vision = nn.Sequential(nn.Conv2d(4, 16, 5, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(128, d), nn.ReLU())
        self.state = nn.Sequential(nn.Linear(STATE_DIM, d), nn.Tanh())
        self.candidate = nn.Sequential(nn.Linear(14, d), nn.ReLU(), nn.Linear(d, d))
        self.fusion = nn.Sequential(nn.Linear(3*d, d), nn.LayerNorm(d), nn.Tanh())
        if cfg.backbone == "transformer":
            layer = nn.TransformerEncoderLayer(d, cfg.heads, 4*d, dropout=0, activation="gelu", batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        elif cfg.backbone == "gru":
            self.temporal = nn.GRU(d, d, cfg.layers, batch_first=True)
        elif cfg.backbone == "ttt":
            self.temporal = FastWeightMemory(d)
        self.query = nn.Linear(d, d)
        self.view_logits = nn.Linear(d, len(VIEWS))
        self.stop_logits = nn.Linear(d, 1)
        self.value = nn.Sequential(nn.Linear(d + (PRIV_DIM if cfg.privileged_critic else 0), d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, observation, history_valid, privileged=None):
        image, state, candidates = [observation[k] for k in ("image", "state", "candidates")]
        b, t = state.shape[:2]
        visual = self.vision(image.reshape(b*t, *image.shape[2:])).reshape(b, t, -1)
        cand = self.candidate(candidates)
        cmask = observation["action_mask"][..., len(VIEWS):-1].to(cand.dtype)
        pooled = (cand*cmask[..., None]).sum(2)/cmask.sum(2, keepdim=True).clamp_min(1)
        tokens = self.fusion(torch.cat([visual, self.state(state), pooled], -1))
        if self.cfg.backbone == "transformer":
            pos = torch.arange(t, device=tokens.device).float()[:, None]
            div = torch.exp(torch.arange(0, tokens.shape[-1], 2, device=tokens.device)*(-math.log(10000.)/tokens.shape[-1]))
            pe = torch.zeros_like(tokens[0])
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*div), torch.cos(pos*div)
            # Right padding: no all-masked prefix queries, no NaN attention.
            causal = torch.ones(t, t, dtype=torch.bool, device=tokens.device).triu(1)
            z = self.temporal(tokens+pe, mask=causal, src_key_padding_mask=~history_valid)
        elif self.cfg.backbone == "gru":
            z, _ = self.temporal(tokens)
        elif self.cfg.backbone == "ttt":
            z = self.temporal(tokens, history_valid)
        else:
            z = tokens
        last = history_valid.long().sum(1)-1
        batch = torch.arange(b, device=z.device)
        hidden, current_candidates = z[batch, last], cand[batch, last]
        scores = (self.query(hidden)[:, None]*current_candidates).sum(-1)/math.sqrt(hidden.shape[-1])
        logits = torch.cat([self.view_logits(hidden), scores, self.stop_logits(hidden)], -1)
        current_mask = observation["action_mask"][batch, last].bool()
        logits = logits.masked_fill(~current_mask, torch.finfo(logits.dtype).min)
        # Actor scores never depend on privileged data; detach value features to
        # prevent privileged critic optimization reshaping the actor backbone.
        value_input = hidden.detach()
        if self.cfg.privileged_critic:
            if privileged is None:
                privileged = hidden.new_zeros(b, PRIV_DIM)
            value_input = torch.cat([value_input, privileged], -1)
        return Categorical(logits=logits), self.value(value_input).squeeze(-1)
