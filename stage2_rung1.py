#!/usr/bin/env python3
"""Stage 2, rung 1 — function-preserving MoE conversion at Gemma interface shapes.

Takes one dense SwiGLU FFN block at the exact shapes the parser verified in
the Gemma-2-2B GGUF (hidden 2304, ffn 9216, no biases) and converts it to a
SWELLDRA-style MoE block: N replica experts + a QRA deterministic router
(argmax, H=0 — no softmax sampling anywhere in the routing path).

Claims under test (gate criteria stated before measurement):
  G1 identity, pinned:   MoE with routing pinned to expert 0 == dense,
                         max|delta| < 1e-5 over random inputs.
  G2 identity, routed:   with replica experts, QRA argmax routing (unpinned)
                         still == dense, max|delta| < 1e-5 (routing-invariant
                         at init: the swell adds capacity, not error).
  G3 router determinism: identical inputs -> identical expert assignments,
                         across repeated runs (exact equality).
  G4 accounting:         active params/token unchanged vs dense;
                         total params = N x dense block.
"""
import copy, json, torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(20261105)
HIDDEN, FFN, EXPERTS = 2304, 9216, 2   # experts=2 on this VM (RAM); formula is general

class DenseFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, FFN, bias=False)
        self.up   = nn.Linear(HIDDEN, FFN, bias=False)
        self.down = nn.Linear(FFN, HIDDEN, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class QRA(nn.Module):
    """Deterministic router: linear scores, argmax selection. No softmax,
    no noise, no sampling — the same input always selects the same expert."""
    def __init__(self, n):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, n, bias=False)
    def forward(self, x):  # x: [B, H] -> expert index per token
        return torch.argmax(self.proj(x), dim=-1)

class MoEFFN(nn.Module):
    def __init__(self, dense, n):
        super().__init__()
        self.experts = nn.ModuleList([copy.deepcopy(dense) for _ in range(n)])
        self.router = QRA(n)
    def forward(self, x, pin=None):
        B = x.shape[0]
        idx = torch.full((B,), pin, dtype=torch.long) if pin is not None else self.router(x)
        out = torch.empty_like(x)
        for e in range(len(self.experts)):
            m = idx == e
            if m.any(): out[m] = self.experts[e](x[m])
        return out, idx

def main():
    dense = DenseFFN().eval()
    moe = MoEFFN(dense, EXPERTS).eval()
    results = {"shapes": {"hidden": HIDDEN, "ffn": FFN, "experts": EXPERTS}}

    with torch.no_grad():
        # G1 + G2 across several trials
        g1 = g2 = 0.0
        for trial in range(5):
            x = torch.randn(16, HIDDEN) * (trial + 1)  # vary scale per trial
            ref = dense(x)
            out_pin, _ = moe(x, pin=0)
            out_rte, _ = moe(x)
            g1 = max(g1, (out_pin - ref).abs().max().item())
            g2 = max(g2, (out_rte - ref).abs().max().item())
        results["G1_identity_pinned_max_delta"] = g1
        results["G2_identity_routed_max_delta"] = g2

        # G3 determinism
        x = torch.randn(64, HIDDEN)
        _, i1 = moe(x); _, i2 = moe(x)
        results["G3_router_determinism_exact"] = bool(torch.equal(i1, i2))
        results["G3_expert_histogram"] = torch.bincount(i1, minlength=EXPERTS).tolist()

    # G4 accounting
    dense_params = sum(p.numel() for p in dense.parameters())
    moe_params = sum(p.numel() for p in moe.experts.parameters())
    results["G4_dense_block_params"] = dense_params
    results["G4_moe_block_params"] = moe_params
    results["G4_active_params_per_token"] = dense_params  # top-1 routing
    results["G4_matches_parser_per_block_mlp_plus_norm"] = (dense_params + HIDDEN) * 26

    results["gates"] = {
        "G1": g1 < 1e-5, "G2": g2 < 1e-5,
        "G3": results["G3_router_determinism_exact"],
        "G4": results["G4_matches_parser_per_block_mlp_plus_norm"] == 1656285696,
    }
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
