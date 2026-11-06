# Stage 1 — Widget Map (Sovereign Composite)

Target topology: **Gemma-2-2B GGUF as parsed by sovereign-gemini-gguf**
(Stage 0, 2026-11-05): 26 blocks, hidden 2304, GQA 8 Q-heads / 4 KV-heads,
head_dim 288, FFN 9216 (SwiGLU), ctx 8192, tied embeddings, 288 tensors,
2,614,341,888 parameters. Parser verdict: graph zero-sorry valid.

Per-block tensor set (×26): `attn_norm`, `attn_q`, `attn_k`, `attn_v`,
`attn_output`, `post_attention_norm`, `ffn_norm`, `ffn_gate`, `ffn_up`,
`ffn_down`, `post_ffw_norm`. Bookends: `token_embd.weight` (tied in/out),
`output_norm.weight`.

## Attachment map — 100% of blocks

| Widget | Attaches at | Status | Adapter / notes |
|---|---|---|---|
| K3 fused attention (GEMM + online softmax, WMMA FP16) | Every block's attention span, between `attn_q/k/v` and `attn_output` | **needs-adapter** | Interface shapes align (Q 2304, K/V 1152 per block). Two gaps: (1) K3 kernels were written for head_dim 128 — Gemma-2 head_dim is 288 (= 16×18, WMMA-tileable, but tiling constants must be re-derived); (2) weights here are Q4_K_M — K3 expects FP16 inputs, so the adapter is a dequant stage (Q4_K_M → FP16) or a Q4-native K3 variant. Adapter I/O: in [seq, 2304] FP16, out [seq, 2304] FP16 |
| K3 GEMM (FFN) | Every block's `ffn_gate`/`ffn_up` (2304→9216) and `ffn_down` (9216→2304) | **needs-adapter** | Same dequant adapter as attention; GEMM shapes are generic. SwiGLU activation already matches the MiMo/K3 design |
| QRA deterministic router | No dense-block attachment exists | **train-new (Stage 2)** | Gemma-2 is dense — there is no router to replace. QRA enters when Stage 2 converts selected FFN blocks to SWELLDRA MoE: router input [2304], output = expert selection over N experts, H=0 deterministic |
| SWELLDRA MoE conversion | Selected blocks' FFN span (Stage 2 candidate: blocks 13–25, upper half) | **train-new (Stage 2)** | Each expert = gate/up/down slice at 2304×9216 (full-size) or 2304×4608 (half-width ×2 experts); function-preserving init copies the dense FFN into expert 0 |
| Meridian time-series head | Parallel head on final hidden states, after `output_norm` | **needs-adapter + Stage 3 distillation** | Projection adapter 2304 → Meridian covariate dims; trained by TimesFM distillation (blueprint Stage 3), not by transplant |
| Sovereign MUM transition audit | Generation loop, outside all blocks | **shape-compatible** | Operates on atoms/states, not tensors: each decoded step is admitted as R(A_n, W) → A_{n+1} or rejected; wraps llama.cpp/K3 decode loop at the harness level |
| Cross-encoder judge (bert-agent, DeBERTa-v3) | Serving path, per (source, claim) pair | **shape-compatible** | External service (Rust daemon); no tensor attachment; consumes text pairs, returns entailment verdict + BLAKE3 attestation |
| ERE P1–P5 gates | Serving path, pre/post generation | **shape-compatible** | External (chat harness / MiMoCore); text-level |
| KID-8B/8K + 6502 SAT kernel | Boot layer, beneath the runtime | **shape-compatible** | Runs on the owner's 6502 line; gates runtime start, independent of model tensors |
| Macro VM control plane | Above the model, capability boundary | **shape-compatible** | Elixir core today; JVM/WASM edge via perplexity-macro-vm/java-sdk |

## Coverage statement

All 26 blocks are covered by the K3 attention + K3 GEMM rows (uniform per
block). Bookend tensors are covered by the Meridian row (output side) and the
MUM row (generation loop). Every needs-adapter item names its I/O shapes;
every train-new item is assigned to its blueprint stage. No block is
unaccounted for.

## Findings carried forward

1. head_dim 288 is the single most unusual shape in this topology — it drives
   the K3 tiling re-derivation and should be a checklist item in Stage 2.
2. Weight tying (one tensor as embedding + output head) means any Stage 2
   vocabulary surgery touches both ends at once.
3. Gemma-2's four-norms-per-block sandwich (pre + post) differs from the
   MiMo spec's two — widget code must not assume MiMo's norm layout.
