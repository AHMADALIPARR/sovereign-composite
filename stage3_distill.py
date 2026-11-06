#!/usr/bin/env python3
"""Stage 3 — TimesFM-3 teacher -> Meridian student distillation (local rung).

Teacher: TimesFM3Evaluator from the owner's Google_Black_Box fork, loaded
from the local google/timesfm-3.0-pytorch checkpoint. The teacher is used
through predict_batch only; swapping the teacher (bigger run, other
checkpoint) touches nothing else in this file.

Student ("Meridian rung-1"): patch forecaster with RevIN instance
normalization (the design language of the fork's cpm_revin_refine):
context -> per-series normalize -> patches of 16 -> linear embed ->
2-layer transformer encoder (d=128) -> linear head -> horizon, then
de-normalize. Trained ONLY on teacher forecasts (distillation); the frozen
ground-truth futures are touched exclusively by the evaluator.

Stated bound (before measurement): on the 32 held-out frozen series,
student truth-MAE <= 1.25 x teacher truth-MAE.
Local slice: context 256, horizon 32 (the frozen file's full spec is
512/128 for the big-machine run; same file, same hash).
"""
import json, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/hatch/workspace/Google_Black_Box/src")
CTX, HOR = 256, 32
N_TEACH_TRAIN = 64
CKPT_DIR = "/home/hatch/workspace/models/timesfm-3.0"
DATA = "/home/hatch/workspace/sovereign-composite/frozen_series.npz"

# ---------------- teacher ----------------
def load_teacher():
    from timesfm3.evaluator import TimesFM3Evaluator
    return TimesFM3Evaluator(checkpoint_path=CKPT_DIR)

def teacher_forecast(ev, contexts, horizon):
    outs = ev.predict_batch([np.asarray(c, dtype=np.float64) for c in contexts], horizon=horizon)
    return np.stack([o.forecast for o in outs]).astype(np.float32)

# ---------------- student ----------------
class MeridianStudent(nn.Module):
    def __init__(self, ctx=CTX, hor=HOR, patch=16, d=128, layers=2, heads=4):
        super().__init__()
        self.patch, self.hor = patch, hor
        self.npatch = ctx // patch
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.zeros(1, self.npatch, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d * self.npatch, hor)
    def forward(self, x):  # x: [B, CTX] raw
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True) + 1e-5
        z = (x - mu) / sd                                   # RevIN
        p = z.view(x.shape[0], self.npatch, self.patch)
        h = self.enc(self.embed(p) + self.pos)
        return self.head(h.flatten(1)) * sd + mu            # de-normalize

def main():
    torch.manual_seed(7)
    d = np.load(DATA)
    tr_c = d["train_contexts"][:N_TEACH_TRAIN, :CTX]
    tr_f = d["train_futures"][:N_TEACH_TRAIN, :HOR]
    ev_c = d["eval_contexts"][:, :CTX]
    ev_f = d["eval_futures"][:, :HOR]

    t0 = time.time()
    ev = load_teacher()
    print(f"teacher loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    teach_tr = teacher_forecast(ev, tr_c, HOR)
    teach_ev = teacher_forecast(ev, ev_c, HOR)
    print(f"teacher forecasts (train {len(tr_c)} + eval {len(ev_c)}) in {time.time()-t0:.1f}s", flush=True)
    np.savez_compressed("/home/hatch/workspace/sovereign-composite/teacher_outputs.npz",
                        teach_train=teach_tr, teach_eval=teach_ev)

    student = MeridianStudent()
    n_params = sum(p.numel() for p in student.parameters())
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3, weight_decay=1e-4)
    X = torch.tensor(tr_c); Y = torch.tensor(teach_tr)
    t0 = time.time()
    for epoch in range(300):
        student.train(); opt.zero_grad()
        loss = ((student(X) - Y) ** 2).mean()
        loss.backward(); opt.step()
    print(f"student trained in {time.time()-t0:.1f}s; final distill MSE {loss.item():.4f}", flush=True)

    student.eval()
    with torch.no_grad():
        stu_ev = student(torch.tensor(ev_c)).numpy()
    mae = lambda a, b: float(np.abs(a - b).mean())
    res = {
        "student_params": n_params,
        "teacher_train_series": len(tr_c),
        "eval_series": len(ev_c),
        "context": CTX, "horizon": HOR,
        "teacher_truth_MAE": mae(teach_ev, ev_f),
        "student_truth_MAE": mae(stu_ev, ev_f),
        "student_teacher_MAE": mae(stu_ev, teach_ev),
        "bound": "student_truth_MAE <= 1.25 * teacher_truth_MAE",
    }
    res["ratio"] = res["student_truth_MAE"] / res["teacher_truth_MAE"]
    res["PASS"] = bool(res["ratio"] <= 1.25)
    print(json.dumps(res, indent=2), flush=True)
    with open("/home/hatch/workspace/sovereign-composite/stage3-results.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
