# Multimodal Ingestion Spec — Native Fusion Pipeline

Source: the owner, 2026-11-05 19:02 PST (his algorithm, pasted in chat).
Two equations were dropped in transmission (after "multiply it by a learned
linear projection matrix E_v:" and in §3/§4 where the rendered formulas are
missing). Those two are restored below and marked [reconstructed]; everything
else is his text, structured.

## Problem statement (owner)

To get a video frame and a text string into the same mathematical space, you
brutally compress a 4D spacetime tensor into a 1D sequence of vectors that
exactly match the dimension of the text embeddings. With hidden dimension
d_model (e.g. 4096), every word, pixel patch, and audio waveform must
eventually become a vector in R^{d_model}.

## 1. Raw tensors

- Text: tokens W = [w_1, …, w_L]; embedding matrix maps them to
  X_text ∈ R^{L × d_model}.
- Video: 4D tensor V ∈ R^{T × H × W × C} — T frames, H×W spatial, C channels (3).

## 2. Spatiotemporal tubelet projection (the slicing)

Raw pixels cannot be fed directly — O(N²) attention OOMs. Slice the video
into tubelets of t (frames) × p (height) × p (width):

  N_v = (T/t) × (H/p) × (W/p) patches.

Each tubelet flattens to x_i ∈ R^{t·p²·C}. Project into latent space:
  [reconstructed] z_i = x_i E_v, with E_v ∈ R^{(t·p²·C) × d_model}.

Result: Z_video = [z_1, …, z_{N_v}], each z_i ∈ R^{d_model} — a patch of
video now identical in shape to a word token.

## 3. Spacetime coordinates (positional encoding)

Video tokens carry no inherent position or time. Add 3D positional
embeddings (each in R^{d_model}):
  [reconstructed] ẑ_i = z_i + e^{time}_{τ(i)} + e^{row}_{ρ(i)} + e^{col}_{κ(i)}.

The tensor now holds the visual data and its exact spacetime coordinates.

## 4. Sequence concatenation (the fusion)

Concatenate along the context dimension:
  [reconstructed] S = [X_text ; Z_video] ∈ R^{(L + N_v) × d_model}.

Optionally add a learned modality embedding (E_txt to text tokens, E_vid to
video tokens) so the model can differentiate sources.

## 5. Cross-modality attention

One self-attention over the union sequence. Q from a text token interacts
with K of a video patch; a high Q_text · K_video^T score is the learned
alignment between a word's semantics and the gradient shapes/colors inside a
specific patch at a specific frame. The softmax routes contextual weight
across modalities natively — no stitching, no side channels.

## Cost law (owner)

Native fusion is VRAM-hungry: 256 video tokens/frame × 1,000 frames =
256,000 context tokens before one word of text is generated — and attention
cost grows as (L + N_v)². This is the wall the agentic/scrubbing ingestion
(select frames on demand rather than ingesting all frames) defends against.

## Composite hooks

- d_model equality is a hard interface constraint — the same quantity the
  GGUF parser verifies as `embedding_length` (2304 in the Stage 0 Gemma
  topology; tubelet projection E_v must target the composite's d_model).
- This spec is the ingestion layer for the FEDERATED-SKUNKSLAB photo/video
  training pipeline (blueprint Stage 2 data strategy): a photo is the
  degenerate case T = t (a single tubelet slab); a document page is one
  frame; interleaved sequences alternate slabs and text spans in S.
- Frame selection under a token budget belongs to the macro VM's capability
  layer (deterministic policy decides which frames enter S), not to the
  model — consistent with the composite's control-flow invariant.
