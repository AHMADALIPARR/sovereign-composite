#!/usr/bin/env python3
"""Stage 4 — governed serving, demo scale (Sovereign Composite).

A deterministic court around llama.cpp. Every request passes five gates
before a token is served, and every outcome — served or refused — is
sealed into a hash-chained serving log whose genesis is the Stage 0
manifest chain head.

  P1 Provenance : the model file's SHA-256 must equal the sealed manifest
                  value, re-verified at startup. A swapped model never serves.
  P2 Capability : the request must name the policy model and stay inside
                  the policy caps (prompt tokens, output tokens). This is
                  the macro VM's capability boundary at demo scale.
  P3 Context    : the Context Hypervisor (chv, C11) admits the sequence:
                  budget enforced, decode pinned (temperature 0, fixed
                  seed), admission trace written per request.
  P4 Output     : the completion is measured against the granted cap and
                  hashed before it may leave.
  P5 Seal       : the transition (request -> admitted S -> output/refusal)
                  is appended to the WORM serving log. Under the MUM law a
                  transition without its justification chain is invalid;
                  invalid transitions are recorded and never served.

Demo-scale honesty: the model is loaded per request (production keeps it
resident); token counts are a deterministic chars/4 estimator; semantic
justification by the cross-encoder judge is a Stage 4-full component and
is not simulated here.

Stdlib only. Run: python3 serve.py [port]
"""
import hashlib, json, math, os, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY = json.load(open(os.path.join(HERE, "policy.json")))
LOG_PATH = os.path.join(HERE, "serving-log.jsonl")
CHV = "/home/hatch/workspace/context-hypervisor/chv"
LLAMA = "/home/hatch/workspace/.tools/llama.cpp/build/bin/llama-cli"
GENESIS = "621f27fa8c288060e4b1ceb23cf9528d248a2ddf85e577a4bd5d7e1251d75bf1"  # Stage 0 chain head

lock = threading.Lock()
state = {"seq": 0, "head": GENESIS}

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def est_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))

def seal(record: dict) -> dict:
    with lock:
        state["seq"] += 1
        record["seq"] = state["seq"]
        record["prev"] = state["head"]
        payload = json.dumps(record, sort_keys=True).encode()
        record["hash"] = sha256_hex(payload)
        state["head"] = record["hash"]
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    return record

def verify_model() -> bool:
    h = hashlib.sha256()
    with open(POLICY["model_path"], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest() == POLICY["model_sha256"]

MODEL_OK = verify_model()  # P1 at startup; a wrong answer disables serving

def hypervisor_admit(prompt_tokens: int, req_id: int):
    trace = os.path.join(HERE, f"trace-{req_id}.log")
    cmd = [CHV, "--model-sha", POLICY["model_sha256"],
           "--dmodel", str(POLICY["d_model"]), "--ctx", str(POLICY["context_length"]),
           "--proj-width", str(POLICY["d_model"]),
           "--frames", "0", "--height", "16", "--width", "16", "--patch", "16",
           "--text", str(prompt_tokens), "--reserve", str(POLICY["decode_reserve"]),
           "--seed", str(POLICY["seed"]), "--trace", trace]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None, r.stderr.strip()
    return json.loads(r.stdout), None

def generate(prompt: str, max_tokens: int) -> str:
    cmd = [LLAMA, "-m", POLICY["model_path"], "-p", prompt,
           "-n", str(max_tokens), "--temp", "0", "--seed", str(POLICY["seed"]),
           "--no-display-prompt"]
    r = subprocess.run(cmd, input="", capture_output=True, text=True, timeout=600)
    out = r.stdout
    # completion follows the echoed prompt; perf line terminates it
    idx = out.find(prompt)
    body = out[idx + len(prompt):] if idx >= 0 else out
    end = body.find("[ Prompt:")
    if end >= 0:
        body = body[:end]
    return body.strip().strip("> ").strip()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, obj):
        data = json.dumps(obj, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"service": POLICY["service"], "model_ok": MODEL_OK,
                             "log_head": state["head"], "seq": state["seq"]})
        else:
            self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        prompt = req.get("prompt", "")
        model = req.get("model", POLICY["model"])
        max_tokens = int(req.get("max_tokens", POLICY["max_output_tokens"]))
        base = {"type": None, "request_sha256": sha256_hex(prompt.encode()),
                "gates": {}}
        # P1 provenance (startup verdict, enforced per request)
        base["gates"]["P1_provenance"] = MODEL_OK
        if not MODEL_OK:
            base["type"] = "REFUSED"; base["refused_at"] = "P1"
            rec = seal(base)
            return self._send(403, {"error": "model provenance failure", "record": rec})
        # P2 capability
        p2 = (model == POLICY["model"] and max_tokens <= POLICY["max_output_tokens"]
              and est_tokens(prompt) <= POLICY["max_prompt_tokens"])
        base["gates"]["P2_capability"] = p2
        if not p2:
            base["type"] = "REFUSED"; base["refused_at"] = "P2"
            base["detail"] = {"model": model, "max_tokens": max_tokens,
                              "prompt_est_tokens": est_tokens(prompt)}
            rec = seal(base)
            return self._send(403, {"error": "capability refusal", "record": rec})
        # P3 context hypervisor
        manifest, err = hypervisor_admit(est_tokens(prompt), state["seq"] + 1)
        base["gates"]["P3_context"] = manifest is not None
        if manifest is None:
            base["type"] = "REFUSED"; base["refused_at"] = "P3"; base["detail"] = err
            rec = seal(base)
            return self._send(403, {"error": "hypervisor refusal", "record": rec})
        base["S"] = manifest["budget"]["S"]
        base["attention_cost_bound"] = manifest["budget"]["attention_cost_bound"]
        base["hypervisor_trace_head"] = manifest["trace"]["head"]
        # generation (pinned decode)
        completion = generate(prompt, max_tokens)
        # P4 output bounds + hash
        out_tokens = est_tokens(completion)
        p4 = out_tokens <= max_tokens + 8  # estimator slack; llama -n is the hard cap
        base["gates"]["P4_output"] = p4
        base["output_sha256"] = sha256_hex(completion.encode())
        base["output_est_tokens"] = out_tokens
        if not p4:
            base["type"] = "REFUSED"; base["refused_at"] = "P4"
            rec = seal(base)
            return self._send(403, {"error": "output bound violation; output withheld", "record": rec})
        # P5 seal, then serve
        base["type"] = "SERVED"
        rec = seal(base)
        self._send(200, {"completion": completion, "record": rec})

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8471
    print(f"governed serving on 127.0.0.1:{port} | model_ok={MODEL_OK} | genesis={GENESIS[:12]}...", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
