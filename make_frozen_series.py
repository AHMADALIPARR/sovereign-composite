#!/usr/bin/env python3
"""Frozen benchmark series for Stage 3 distillation (Sovereign Composite).

Deterministic generator: five series families, fixed seeds, no external
data. One file serves both the local rung (sliced: context 256, horizon 32)
and the big-machine run (full: context 512, horizon 128).

Families:
  A trend + daily seasonality + noise
  B trend + weekly seasonality + mid-series level shift
  C stationary AR(2)
  D seasonality + sparse spikes
  E random walk with drift
Output: frozen_series.npz + frozen_series.spec.json (with SHA-256 of the npz
recorded by the sealing step, not here).
"""
import json
import numpy as np

CTX, HOR = 512, 128
N_TRAIN, N_EVAL = 256, 32
BASE_SEED = 20261105

def gen_series(rng, family, n):
    t = np.arange(n, dtype=np.float64)
    if family == "A":
        trend = rng.uniform(0.005, 0.03) * t
        seas = rng.uniform(2.0, 5.0) * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 6.28))
        noise = rng.normal(0, rng.uniform(0.3, 0.8), n)
        return (10 + trend + seas + noise).astype(np.float32)
    if family == "B":
        trend = rng.uniform(-0.01, 0.02) * t
        seas = rng.uniform(1.5, 4.0) * np.sin(2 * np.pi * t / 168 + rng.uniform(0, 6.28))
        shift = np.where(t >= n // 2, rng.uniform(3.0, 8.0), 0.0)
        noise = rng.normal(0, rng.uniform(0.3, 0.7), n)
        return (20 + trend + seas + shift + noise).astype(np.float32)
    if family == "C":
        x = np.zeros(n); e = rng.normal(0, 1.0, n)
        phi1, phi2 = rng.uniform(0.4, 0.7), rng.uniform(0.1, 0.3)
        for i in range(2, n):
            x[i] = phi1 * x[i-1] - phi2 * x[i-2] + e[i]
        return (x + 15).astype(np.float32)
    if family == "D":
        seas = rng.uniform(2.0, 6.0) * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 6.28))
        spikes = np.zeros(n)
        idx = rng.choice(n, size=max(2, n // 64), replace=False)
        spikes[idx] = rng.uniform(4.0, 10.0, len(idx))
        noise = rng.normal(0, rng.uniform(0.2, 0.5), n)
        return (12 + seas + spikes + noise).astype(np.float32)
    # E: random walk with drift
    steps = rng.normal(rng.uniform(0.005, 0.03), rng.uniform(0.3, 0.7), n)
    return (30 + np.cumsum(steps)).astype(np.float32)

def build(count, seed0):
    fams = ["A", "B", "C", "D", "E"]
    ctxs, futs, fam_out = [], [], []
    for i in range(count):
        rng = np.random.default_rng(seed0 + i)
        fam = fams[i % len(fams)]
        s = gen_series(rng, fam, CTX + HOR)
        ctxs.append(s[:CTX]); futs.append(s[CTX:]); fam_out.append(fam)
    return np.stack(ctxs), np.stack(futs), np.array(fam_out)

def main():
    tr_c, tr_f, tr_fam = build(N_TRAIN, BASE_SEED)
    ev_c, ev_f, ev_fam = build(N_EVAL, BASE_SEED + 100000)
    np.savez_compressed(
        "/home/hatch/workspace/sovereign-composite/frozen_series.npz",
        train_contexts=tr_c, train_futures=tr_f, train_families=tr_fam,
        eval_contexts=ev_c, eval_futures=ev_f, eval_families=ev_fam)
    spec = {"context": CTX, "horizon": HOR, "n_train": N_TRAIN, "n_eval": N_EVAL,
            "base_seed": BASE_SEED, "families": ["A", "B", "C", "D", "E"],
            "local_slice": {"context": 256, "horizon": 32}}
    with open("/home/hatch/workspace/sovereign-composite/frozen_series.spec.json", "w") as f:
        json.dump(spec, f, indent=2)
    print("frozen series written:", tr_c.shape, ev_c.shape)

if __name__ == "__main__":
    main()
