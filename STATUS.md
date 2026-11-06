# STATUS — Sovereign Composite

## Verified (2026-11-05, this VM: 4 vCPU, no GPU)

- Stage 0: both GGUFs byte-exact downloads; parser inventories, Graph IR,
  and zero-sorry validation for two architectures (gemma2, nemotron);
  CPU inference transcript recorded.
- Context Hypervisor: self-test (FIPS SHA-256 vectors), budget scenario,
  photo scenario, width-2305 refusal, byte-identical rerun traces.
- Stage 2 rung 1: identity max |Δ| = 0.0 at 2 experts (VM RAM); accounting
  reconciles with the parser total exactly.
- Stage 3 local rung: PASS at slice scale (context 256, horizon 32,
  64 teaching series). Ratio 1.061 against the 1.25 bound.
- Stage 4 demo: server ran; P1 startup re-hash matched; two refusals and
  one served request sealed; serving-log chain independently recomputed
  valid. Server stopped after the demo.

## Limits

- Serving loads the model per request and estimates tokens as chars/4.
  Production serving keeps the model resident; neither changes the gate
  semantics.
- Stage 3's teacher is Google's TimesFM-3.0 checkpoint run through the
  owner's fork's Evaluator; the fork's timesfm3 sources carry Google
  copyright headers (Apache-2.0) and are not vendored into this repo.
- The parser and hypervisor consume GGUF + topology only; no vision
  encoder is wired yet — visual tokens are governed as accounting and
  admission, per the ingestion spec.

## Open

- Stage 2 rung 2: Q4_K dequantizer and transplant of real GGUF weights
  into MoE experts; then a fine-train delta on a corpus.
- Stage 3 full run on the owner's hardware (runbook in repo).
- Stage 4 full: cross-encoder judge inside the loop; resident-model
  serving; `mimo_core.wasm` deployment to the snapkitty-agents portal
  (requires a push: owner go-ahead + token).
- Jetson serving stages (SSH authorization outstanding).
- The composite's final name is the owner's to coin; "Sovereign
  Composite" is the working title.
- Publication (Stage 5) is the owner's call: destination, topics, and
  tag are prepared locally.
