# Blueprint — Sovereign Composite Model

**Working title only; the name is the owner's to coin.**
Drafted 2026-11-05 by Muse (auditor) from the owner's synthesis of the same evening.
Status: EXECUTING — Stage 0/1 in progress under the owner's go-ahead of 2026-11-05 18:32 PST.

---

## 1. Thesis

Assemble one sovereign model from components the owner already built, instead of
downloading a stranger's model and calling it a stack:

- **Eyes** — `sovereign-gemini-gguf`: zero-dependency GGUF parser + Graph IR.
  Reconstructs a black-box model's topology from its binary, weights never loaded.
- **Growth method** — `SWELLDRA-200B`: the swelling ladder. Parameters grow rung by
  rung; a rung is promoted only when measured gates pass.
- **Skeleton** — a Gemma-class decoder-only transformer (GQA + SwiGLU + RMSNorm +
  RoPE), the shape the parser was built by dissecting.
- **Muscle design** — `sovereign-mimo-4b` architecture (Hilbert-4B spec) and the
  `sovereign-kimi` K3 Nano bare-metal CUDA harness + kernels.
- **Nervous system** — `perplexity-macro-vm`: a deterministic 16-bit VM that owns
  control flow. The model proposes; the VM disposes, behind typed capability
  boundaries, every retired instruction traced and hashed.
- **Transition law** — `sovereign-mum` (added by the owner 2026-11-05 18:42 PST;
  "a huge addition"): Google's MUM formalized as a deterministic multimodal
  MDP. The central invariant **R(A_n, W) → A_{n+1}** — a generation step is a
  state transition that must be mathematically justified by the input atoms, or
  it is rejected as invalid. A hallucination is not a quirk; it is an invalid
  state transition. Ships with atom semantics, Lean 4 invariants, a WORM chain
  per transition, the KID-8B/8K child-safety kernel, and a 6502 SAT kernel.
- **Retained weights (in hand today)** — the Nemotron-line GGUFs already published
  under the owner's HF account (`Snapkitty/snapkitty-merged`,
  `snapkitty-nemotron`, `snapkitty-harness`, all Q4_K_M, ~2.7 GB).
- **Teacher** — Google's TimesFM (owner's fork: `Google_Black_Box` / timesfm3),
  parsed the same way, distilled into a time-series capability (the Meridian line).
- **Judge** — `bert-agent`: a DeBERTa-v3 cross-encoder entailment verifier.
  Given a source and a generated claim, it returns a bounded entailment score,
  a verdict, and a BLAKE3 attestation sealed into a WORM chain, through the
  ERE gates (Rust/Tokio daemon, TensorRT FP16, Lean invariants, 11/11 tests).
  It is the mechanical checker for MUM's transition law: MUM declares a step
  must be justified; the cross-encoder is what measures the justification.

The output is a model whose weights, recipe, and verification all live in the
owner's repos under the Sovereign license and a hash-chained manifest — a solid
model, *retained*: nothing essential stranded on a rig in storage, nothing rented.

## 2. Invariants (the contract)

1. **Parse before trust.** No weight file is used until the owner's parser has
   inventoried it: tensor count, shapes, dtypes, offsets, bounds-checked.
2. **Provenance from evidence.** Every artifact carries its source, license, and
   SHA-256 in the stage manifest. Model cards are written from the manifest,
   never from narrative. (Lesson of the MiMo card, 2026-11-05.)
3. **Shape-align or train — never pour.** Foreign weights initialize a slot only
   where name, shape, and dtype align exactly. Everything else is trained or
   distilled. No silent reshaping.
4. **Gates are measured.** A swell rung is promoted only on recorded metrics.
   SWELLDRA's existing gate semantics stand: OCR ≥ 0.85, SQL exec ≥ 0.70,
   graph ≥ 0.80, logic ≥ 0.95 (minimums), router load ≤ 2.0 (maximum).
5. **The VM owns control flow.** The model never calls tools directly; it
   requests capabilities from the macro VM, which accepts or refuses and seals
   the transcript.
5a. **Transitions must be justified (Sovereign MUM).** Every generation step is
   a transition R(A_n, W) → A_{n+1} audited before A_{n+2} proceeds. A step
   that cannot be justified from the input atoms is invalid and is rejected,
   not emitted. The audit result joins the WORM chain per transition.
6. **ERE before output.** Every user-visible response passes the ERE P1–P5
   gates (secrets, injection, loop, telemetry, seal) as implemented in
   `sovereign-chat-harness` / MiMoCore.
7. **Zero paid dependencies.** No vendor APIs, no rented compute in the critical
   path. Public weights are used only under their own licenses, recorded per file.
8. **Seal every stage.** Each stage ends with a WORM-style manifest: inputs,
   outputs, hashes, tool versions, verdict. A stage without a manifest is not done.

## 3. Component inventory (verified 2026-11-05)

| Component | Location | State |
|---|---|---|
| GGUF parser + Graph IR | `~/workspace/sovereign-gemini-gguf` @ 6389643 | Code complete; 8/8 tests per its report; built by parsing a 15.48 GB, 291-tensor, 36-block Gemini-class GGUF |
| SWELLDRA | `~/workspace/swelldra-200b` @ c31b037 | **Runs** (verified 2026-11-05): 17.6M-param MoE constructs, forward pass executes, 4/4 capability interface benchmarks pass. `generate()` stub; no training loop; no weights |
| MiMo-4B architecture | `~/workspace/sovereign-mimo-4b` @ bc634ae | Spec + pruning scripts + kernels complete. **Weights exist only on bbqbaddie (storage)** — pruned reward→instruct checkpoint, kept local, never uploaded |
| Kimi K3 Nano | `~/workspace/sovereign-kimi` @ e86bc51 (+ copy in mimo repo) | Single-file CUDA harness, K3M1 `model.bin` format, targets sm_86; Butterfly Triple-Lock Forge incl. Why3 proof |
| Macro VM | `~/workspace/perplexity-macro-vm` @ c12f4a4 | Elixir/OTP VM + Swift console + Java SDK; live Pages demo |
| Sovereign MUM | `~/workspace/sovereign-mum` @ 82af086 | Formal transition layer: R(A_n, W) → A_{n+1} invariant, atom semantics (`src/mum/atom.py`), KID-8B/8K safety kernel, 6502 SAT kernel (asm); README claims Lean 4 invariants (paper-linked). License history: shipped with an agent-substituted tri-license (AGPL/BSL/MIT, attributed by the owner to Claude); **restored to Sovereign Source License v1.0 locally 2026-11-05 @ c254930 — push outstanding (needs owner token)** |
| BERT cross-encoder (bert-agent) | `~/workspace/bert-agent` @ 8165760 | DeBERTa-v3-base cross-encoder entailment verifier (ANLI + TrueTeacher + MNLI), calibrate/train/export code, ERE gate, Rust daemon (TensorRT FP16, WORM ledger); 11/11 tests per badge; no weights file posted (base is public microsoft/deberta-v3-base). License history: same agent-substituted tri-license as sovereign-mum; **restored to Sovereign Source License v1.0 locally 2026-11-05 @ 64d6272 — push outstanding (needs owner token)** |
| Chat harness | `~/workspace/sovereign-chat-harness` @ a812ff9 | ERE P1–P5 gating, Gradio Space front end |
| Agent home | `~/workspace/snapkitty-agents` @ 4610e78 | Pages portal LIVE (verified 200); talks to Ollama localhost:11434; `mimo_core.wasm` 404 (source present, build not deployed) |
| TimesFM fork | `~/workspace/Google_Black_Box` @ aa48015 | Upstream TimesFM + owner's timesfm3 (RevIN refine, evaluator, forecaster, tests) |
| Retained weights | HF `Snapkitty/*` | 3 GGUFs only: snapkitty-merged (2,705.9 MB, 38 dl), snapkitty-nemotron (2,697.4 MB), snapkitty-harness (2,697.4 MB). Nemotron Mini 4.2B line, NVIDIA Community Model License |

## 4. The composite architecture

```
                 ┌─────────────────────────────────────────────┐
   user ───────► │  Perplexity Macro VM (control plane)        │
                 │  deterministic · capability gates · traced  │
                 └──────────────┬──────────────────────────────┘
                                │ typed evidence / requests
                 ┌──────────────▼──────────────────────────────┐
                 │  Sovereign MUM transition audit             │
                 │  R(A_n, W) → A_{n+1} justified, or rejected │
                 ├─────────────────────────────────────────────┤
                 │  ERE P1–P5 gates (chat harness / MiMoCore)  │
                 └──────────────┬──────────────────────────────┘
                                │
        ┌───────────────────────▼────────────────────────┐
        │  Composite model (Gemma-class skeleton)        │
        │  GQA · SwiGLU · RMSNorm · RoPE                 │
        │  widgets: QRA router · K3 kernels · Meridian   │
        │  experts (SWELLDRA MoE slots):                 │
        │    language · code · time-series (TimesFM-     │
        │    distilled) · OCR/SQL/Graph/Logic (SWELLDRA) │
        └───────────────────────┬────────────────────────┘
                                │ weights: parsed, aligned, swelled
                 ┌──────────────▼──────────────────────────────┐
                 │  Parser (sovereign-gemini-gguf) — verifies  │
                 │  every artifact in/out; WORM manifest seals │
                 └─────────────────────────────────────────────┘
```

**Widgets** (custom components, and where each attaches):
- **QRA deterministic router** (from sovereign-mimo-4b / sovereign-qra): replaces
  softmax routing in MoE layers; H=0 deterministic selection.
- **K3 kernels** (sovereign-kimi): fused GEMM + online-softmax attention, WMMA
  FP16; CUDA path for GPU serving, CPU path for verification runs.
- **Meridian time-series block** (sovereign-gemini-gguf/meridian + timesfm3):
  covariate-aware forecasting head; the TimesFM teacher's distilled form.
- **Sovereign MUM transition audit**: wraps generation itself — each step is
  admitted only if justified from the input atoms; unjustified steps are
  rejected before emission and the rejection is sealed. Sits between the
  control plane and the model, above the ERE content gates.
- **Cross-encoder judge** (bert-agent, DeBERTa-v3): scores (source, claim)
  pairs for entailment at every MUM transition checkpoint and at the end of
  every retrieval pipeline; verdict + BLAKE3 attestation sealed in the WORM
  chain; a claim that is not entailed does not propagate.
- **KID-8B/8K + 6502 SAT kernel** (sovereign-mum): the boot-critical safety
  layer — a child-safety kernel small enough to audit completely, running on
  the owner's 6502 line (cf. ace-sovereign machine, Miami Vice runtime).
- **ERE gates**: outside the model, in the serving path — non-negotiable placement.
- **SWELLDRA capability engines**: OCR / SQL / Graph / Logic verifiers as expert
  scorers; their accuracies are the ladder's gate metrics.

## 5. Weights strategy

| Source | License | Role |
|---|---|---|
| snapkitty-merged Q4_K_M (HF, owner's) | NVIDIA Community Model License | First retained body: parse, verify, run locally in Stage 0 |
| Gemma-class GGUF (public, e.g. ggml-org builds) | Gemma Terms of Use | The skeleton reference: parser inventory + Graph IR of the target shape |
| TimesFM checkpoint (public) | Apache-2.0 | Teacher for the time-series expert (distillation, not transplant) |
| MiMo-7B (public) | Apache-2.0 (Xiaomi) | The original pruning base — regeneration path for MiMo-4B weights if the owner elects it; NOT required by this blueprint |
| SWELLDRA-trained | Sovereign | Each swell rung's delta: trained here, sealed here |

Rules: per-file license recorded in the manifest; NVIDIA-licensed weights are
used for local inference/research and are never relicensed; anything published
carries only what its license permits.

**Steering provenance (what safety comes bolted on).** A pretrained base
carries almost no behavioral steering — corpus statistics, plus whatever the
vendor filtered from its data. An *instruct/RLHF* checkpoint additionally
carries the vendor's refusal patterns, hedging, and house style **in the
weights**, applied by SFT + preference tuning; it survives prompting and is
only reduced by further training. Per source: Gemma-2-2B-**it** = Google's
instruction tuning included; snapkitty-merged = NVIDIA's instruct tuning,
partly overwritten by the owner's own merge/fine-tune; TimesFM = no chat
behavior to inherit (a forecaster); DeBERTa = encoder, no generative behavior;
SWELLDRA skeleton = none, it is the owner's. Rule: the manifest records each
source's tuning state (base / SFT / RLHF, and by whom). The composite's
safety is supplied by the owned deterministic layers (MUM, ERE, cross-encoder,
macro VM) — never by inherited refusals — and base checkpoints are preferred
where a clean behavioral slate matters.

## 6. Stages and done-states

**Stage 0 — Ground truth (this run).**
Pull snapkitty-merged + one Gemma-class GGUF to `~/workspace/models/`. Run the
owner's parser over both. *Done when:* parser inventory files exist for both
(tensor count, shapes, total params computed from tensors), SHA-256 of each GGUF
recorded, and one retained model answers a frozen prompt on CPU (llama.cpp)
with the transcript saved. Parser failures are reported as findings, not hidden.

**Stage 1 — Widget map.**
Against the parsed Gemma-class Graph IR, write the attachment map: which blocks
the QRA router, K3 attention, and Meridian head replace or augment; which
SWELLDRA expert slots exist at this scale; every mapping marked
shape-compatible / needs-adapter / train-new. *Done when:* the map covers 100%
of blocks and every "needs-adapter" item names its adapter's input/output shapes.

**Stage 2 — First swell rung.**
Initialize the composite at the parsed scale; swell ONE rung per the SWELLDRA
ladder discipline (width/depth growth with function-preserving initialization
where shapes allow); fine-train the delta on available compute (Jetson-class;
SSH authorization is the owner's one-line gate). *Done when:* the rung's gate
metrics are recorded in the manifest — pass or fail, both are results.

**Stage 3 — Expert distillation.**
Distill TimesFM (teacher) into the Meridian time-series expert; evaluate with
the timesfm3 evaluator against the teacher on frozen series. *Done when:*
forecast error vs teacher within a stated bound, recorded.

**Stage 4 — Governed serving.**
Serve the composite behind the macro VM: model requests capabilities, VM
decides, ERE gates screen output, transcript sealed per session. Deploy the
missing `mimo_core.wasm` to the Pages portal as part of this stage. *Done when:*
an end-to-end session transcript exists with VM trace + ERE verdicts + seal.

**Stage 5 — Publication (owner's call only).**
Repo assembled from stage manifests; model card written FROM the manifests;
push only on the owner's explicit go-ahead with a fresh single-use token,
LICENSE checked first. *Done when:* remote HEAD verified equal to local.

## 7. Compute plan (honest)

- This VM: parsing, verification, CPU inference of ≤4B Q4 models, SWELLDRA
  skeleton runs. No GPU, no CUDA toolchain — K3 harness cannot compile here.
- Jetson Orin NX (16 GB): fits 4B Q4 serving and small fine-tunes; sm_87 retarget
  needed for K3 (Makefile is sm_86). **Blocked on the owner's SSH key install.**
- bbqbaddie (RTX 3080, storage): the original MiMo venue; Stage 2+ accelerates
  when she returns. Not required for Stages 0–1.
- The 200B end of the ladder remains a *direction*, gated rung by rung; this
  blueprint commits to no rung beyond the next one.

## 8. Risks and open items

- Parser was written against a Gemini-arch GGUF; Nemotron (llama-arch) and
  Gemma GGUFs may expose parser gaps. Stage 0 measures this; gaps become
  findings with file/line references.
- NVIDIA Community Model License constrains redistribution of the Nemotron
  weights; the composite's publishable form may need the Gemma/SWELLDRA-trained
  weights instead. Flagged for Stage 5, not a blocker for local stages.
- MiMo-4B's public model card states a provenance the owner's account refines
  (pruned, checkpoint local). Card correction is the owner's call.
- Training data for swell rungs is not yet selected. Stage 2 cannot start its
  training half until the owner names a corpus; its initialization half can.
  Owner's candidate (2026-11-05): the **photo-training pipeline from
  FEDERATED-SKUNKSLAB** (`~/workspace/FEDERATED-SKUNKSLAB` @ 59bec4e) — corpus
  rendered as images and read by vision ingestion (dense structure per visual
  token, layout preserved), with the repo's AgentFishTank frontend as corpus
  visualization and the BRICK/seal layer as corpus provenance. NOTE: the
  published tree carries the seal framework + frontend; the pipeline code
  itself is not in the tree (same pattern as the MiMo weights — local on the
  owner's iron). The "1000× faster" figure is the owner's, unverified.

## 9. Owner decisions outstanding

1. The composite's name.
2. First base for the skeleton: Gemma-class public GGUF (blueprint default) or
   the retained Nemotron body.
3. Jetson SSH authorization (one line, gates Stage 2+ and all Jetson serving).
4. Training corpus for the first swell rung.
5. Whether MiMo-4B weights are regenerated via the prune scripts or left
   sleeping until bbqbaddie returns.


## Addendum (owner, 2026-11-05 ~19:19 PST) — The Context Hypervisor

The macro VM's role over multimodal ingestion is named by the owner: **the
Context Hypervisor**. Frame selection is a capability decision made by the
deterministic layer before any tensor is loaded:

1. **The O(N²) bound is strictly enforced.** The VM mathematically caps N_v
   (video/visual tokens admitted) before VRAM is touched. The model cannot
   probabilistically blow out the context window — contrast cloud "agentic
   scrubbing," where the LLM itself decides what to see, making the token
   cost (L + N_v)² non-deterministic and the reasoning path unauditable.
2. **Zero-trust traceability.** If the model hallucinates or misses visual
   context, there is no guessing whether an internal sub-agent skipped a
   frame: the VM's trace log records exactly which frame coordinates were
   granted entry into sequence S.
3. **The LLM is reduced to its proper role:** a pure function
   f(S) → tokens, over a context it did not choose.

Companion principle (same message): the d_model constraint is not a
configuration flag — it is **the physical width of the reality the model can
perceive**. A tubelet projection emitting a vector of length d_model + 1 does
not degrade; it crashes the multiplication. Enforcing shape invariants in
the parser (sovereign-gemini-gguf) is therefore a **topological admission
check** executed before the GPU spins up: the contract is verified by the
machine.

Decode-side completion (auditor's note, same evening): the hypervisor makes
the *input* side deterministic — context, cost, admission. Full f(S)
determinism additionally requires the decode policy pinned (greedy /
deterministic selection, e.g. QRA-style H=0), which this stack already
leans toward; sampling temperature is the one remaining probabilistic door,
and it belongs to the VM's policy too.
