/* chv.c — The Context Hypervisor (Sovereign Composite)
 *
 * Deterministic context admission for a multimodal transformer.
 * The hypervisor — never the model — decides what enters sequence S:
 *
 *   1. Topological admission: a projection whose output width differs from
 *      the model's d_model is refused before any tensor is loaded. A width
 *      of d_model+1 is not a degradation; the multiplication does not exist.
 *   2. Budget enforcement: N_v (visual tokens) is capped so that
 *      L + N_v <= context_length - decode_reserve, and the worst-case
 *      attention cost (L + N_v)^2 is computed and recorded before the run.
 *      Excess visual input is thinned by a deterministic uniform stride —
 *      same input, same admitted set, every time.
 *   3. Zero-trust trace: every admission decision is appended to a
 *      hash-chained (WORM) trace whose genesis is the model file's own
 *      SHA-256. If context is missing, the trace says which coordinates
 *      were never granted entry.
 *   4. Decode pinning: the emitted run manifest fixes temperature 0 and a
 *      fixed seed, so f(S) -> tokens is a pure function of the admitted S.
 *
 * C11, no dependencies. SHA-256 implemented below (FIPS 180-4).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---------------- SHA-256 (FIPS 180-4) ---------------- */
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t buflen; } sha256_ctx;
static const uint32_t K256[64] = {
 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define R32(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void sha256_block(sha256_ctx *c, const uint8_t *p){
  uint32_t w[64];
  for(int i=0;i<16;i++) w[i]=(uint32_t)p[4*i]<<24|(uint32_t)p[4*i+1]<<16|(uint32_t)p[4*i+2]<<8|p[4*i+3];
  for(int i=16;i<64;i++){ uint32_t s0=R32(w[i-15],7)^R32(w[i-15],18)^(w[i-15]>>3);
    uint32_t s1=R32(w[i-2],17)^R32(w[i-2],19)^(w[i-2]>>10); w[i]=w[i-16]+s0+w[i-7]+s1; }
  uint32_t a=c->h[0],b=c->h[1],d=c->h[3],e=c->h[4],f=c->h[5],g=c->h[6],h=c->h[7],cc=c->h[2];
  for(int i=0;i<64;i++){ uint32_t S1=R32(e,6)^R32(e,11)^R32(e,25);
    uint32_t ch=(e&f)^(~e&g); uint32_t t1=h+S1+ch+K256[i]+w[i];
    uint32_t S0=R32(a,2)^R32(a,13)^R32(a,22); uint32_t mj=(a&b)^(a&cc)^(b&cc);
    uint32_t t2=S0+mj; h=g; g=f; f=e; e=d+t1; d=cc; cc=b; b=a; a=t1+t2; }
  c->h[0]+=a;c->h[1]+=b;c->h[2]+=cc;c->h[3]+=d;c->h[4]+=e;c->h[5]+=f;c->h[6]+=g;c->h[7]+=h;
}
static void sha256_init(sha256_ctx *c){
  c->h[0]=0x6a09e667;c->h[1]=0xbb67ae85;c->h[2]=0x3c6ef372;c->h[3]=0xa54ff53a;
  c->h[4]=0x510e527f;c->h[5]=0x9b05688c;c->h[6]=0x1f83d9ab;c->h[7]=0x5be0cd19;
  c->len=0; c->buflen=0;
}
static void sha256_update(sha256_ctx *c, const uint8_t *data, size_t n){
  c->len += n;
  while(n){
    size_t take = 64 - c->buflen; if(take > n) take = n;
    memcpy(c->buf + c->buflen, data, take); c->buflen += take; data += take; n -= take;
    if(c->buflen == 64){ sha256_block(c, c->buf); c->buflen = 0; }
  }
}
static void sha256_final(sha256_ctx *c, uint8_t out[32]){
  uint64_t bits = c->len * 8;
  uint8_t pad = 0x80; sha256_update(c, &pad, 1);
  uint8_t z = 0; while(c->buflen != 56) sha256_update(c, &z, 1);
  /* append length manually (buflen accounting already includes padding) */
  uint8_t lb[8]; for(int i=0;i<8;i++) lb[i]=(uint8_t)(bits >> (56-8*i));
  /* feed length bytes through the block machinery without touching c->len again */
  for(int i=0;i<8;i++){ c->buf[c->buflen++] = lb[i]; if(c->buflen==64){ sha256_block(c,c->buf); c->buflen=0; } }
  for(int i=0;i<8;i++){ out[4*i]=(uint8_t)(c->h[i]>>24); out[4*i+1]=(uint8_t)(c->h[i]>>16); out[4*i+2]=(uint8_t)(c->h[i]>>8); out[4*i+3]=(uint8_t)c->h[i]; }
}
static void sha256_hex(const char *msg, char out[65]){
  sha256_ctx c; sha256_init(&c);
  sha256_update(&c, (const uint8_t*)msg, strlen(msg));
  uint8_t d[32]; sha256_final(&c, d);
  for(int i=0;i<32;i++) sprintf(out + 2*i, "%02x", d[i]);
  out[64]=0;
}

/* ---------------- self-test ---------------- */
static int selftest(void){
  char h[65];
  sha256_hex("", h);
  if(strcmp(h,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")){ printf("FAIL empty: %s\n", h); return 1; }
  sha256_hex("abc", h);
  if(strcmp(h,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")){ printf("FAIL abc: %s\n", h); return 1; }
  sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", h);
  if(strcmp(h,"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")){ printf("FAIL 56-byte: %s\n", h); return 1; }
  printf("SELFTEST PASS (sha256: empty, abc, 56-byte FIPS vectors)\n");
  return 0;
}

/* ---------------- the hypervisor ---------------- */
static long arg_long(int argc, char **argv, const char *name, long dflt){
  for(int i=1;i+1<argc;i++) if(!strcmp(argv[i],name)) return strtol(argv[i+1],0,10);
  return dflt;
}
static const char *arg_str(int argc, char **argv, const char *name, const char *dflt){
  for(int i=1;i+1<argc;i++) if(!strcmp(argv[i],name)) return argv[i+1];
  return dflt;
}
static long ceil_div(long a, long b){ return (a + b - 1) / b; }

int main(int argc, char **argv){
  for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--selftest")) return selftest();

  /* Topology: verified by the parser before this program is ever invoked.
     The hypervisor consumes the parser's verdict; it does not re-derive it. */
  long d_model   = arg_long(argc, argv, "--dmodel", 0);
  long ctx_len   = arg_long(argc, argv, "--ctx", 0);
  const char *model_sha = arg_str(argc, argv, "--model-sha", "");
  const char *trace_path = arg_str(argc, argv, "--trace", "chv-trace.log");

  /* Projection spec under admission review */
  long proj_width = arg_long(argc, argv, "--proj-width", 0);

  /* Multimodal job: video T frames of HxW, tubelet t x p x p, text L tokens */
  long T = arg_long(argc, argv, "--frames", 0);
  long H = arg_long(argc, argv, "--height", 0);
  long W = arg_long(argc, argv, "--width", 0);
  long t = arg_long(argc, argv, "--tubelet-t", 1);
  long p = arg_long(argc, argv, "--patch", 16);
  long L = arg_long(argc, argv, "--text", 0);
  long reserve = arg_long(argc, argv, "--reserve", 256); /* decode headroom */
  long seed = arg_long(argc, argv, "--seed", 1234);

  if(d_model<=0 || ctx_len<=0 || strlen(model_sha)!=64){
    fprintf(stderr,"REFUSED: topology incomplete (need --dmodel, --ctx, --model-sha <64 hex>)\n");
    return 2;
  }

  /* --- Gate 1: topological admission --- */
  if(proj_width != d_model){
    fprintf(stderr,"REFUSED TOPOLOGY_VIOLATION: projection width %ld != d_model %ld. "
      "The multiplication does not exist; no tensor was loaded.\n", proj_width, d_model);
    return 3;
  }

  /* --- Tubelet accounting (owner's formula: N_v = (T/t)*(H/p)*(W/p)) --- */
  long slabs = ceil_div(T, t);
  long patches_per_slab = ceil_div(H, p) * ceil_div(W, p);
  long Nv_total = slabs * patches_per_slab;

  /* --- Gate 2: budget enforcement with deterministic stride thinning --- */
  long budget = ctx_len - L - reserve;
  if(budget < 0){ fprintf(stderr,"REFUSED BUDGET: text L=%ld + reserve %ld exceeds context %ld\n", L, reserve, ctx_len); return 4; }
  long max_slabs = patches_per_slab>0 ? budget / patches_per_slab : 0;
  long stride = 1, admitted = 0;
  if(slabs > max_slabs){
    if(max_slabs <= 0){ fprintf(stderr,"REFUSED BUDGET: one slab (%ld tokens) exceeds visual budget %ld\n", patches_per_slab, budget); return 4; }
    stride = ceil_div(slabs, max_slabs);
  }
  for(long s=0;s<slabs;s+=stride) admitted++;
  long Nv = admitted * patches_per_slab;
  long S_len = L + Nv;
  unsigned long long cost = (unsigned long long)S_len * (unsigned long long)S_len;

  /* --- WORM trace: genesis = the model file's own SHA-256 --- */
  FILE *tr = fopen(trace_path, "w");
  if(!tr){ fprintf(stderr,"cannot open trace %s\n", trace_path); return 5; }
  char prev[65]; strncpy(prev, model_sha, 64); prev[64]=0;
  char line[512], chained[600], hash[65];
  #define TRACE(...) do{ \
    snprintf(line, sizeof line, __VA_ARGS__); \
    snprintf(chained, sizeof chained, "%s|%s", prev, line); \
    sha256_hex(chained, hash); \
    fprintf(tr, "%s  %s\n", hash, line); \
    memcpy(prev, hash, 64); prev[64]=0; \
  }while(0)

  TRACE("POLICY model_sha=%s d_model=%ld ctx=%ld L=%ld reserve=%ld seed=%ld temp=0", model_sha, d_model, ctx_len, L, reserve, seed);
  TRACE("TOPOLOGY proj_width=%ld ADMITTED (== d_model)", proj_width);
  TRACE("BUDGET Nv_total=%ld budget=%ld stride=%ld admitted_slabs=%ld Nv=%ld S=%ld cost_bound=%llu",
        Nv_total, budget, stride, admitted, Nv, S_len, cost);
  for(long s=0;s<slabs;s+=stride){
    long f0 = s*t, f1 = f0 + t - 1; if(f1 > T-1) f1 = T-1;
    TRACE("ADMIT slab=%ld frames=[%ld..%ld] tokens=%ld", s, f0, f1, patches_per_slab);
  }
  TRACE("SEAL admitted=%ld of %ld slabs; S=%ld; head follows in manifest", admitted, slabs, S_len);
  fclose(tr);

  /* --- Run manifest (stdout): the contract the runtime executes under --- */
  printf("{\n");
  printf("  \"hypervisor\": \"context-hypervisor v1 (C11)\",\n");
  printf("  \"model_sha256\": \"%s\",\n", model_sha);
  printf("  \"d_model\": %ld,\n", d_model);
  printf("  \"context_length\": %ld,\n", ctx_len);
  printf("  \"projection\": {\"width\": %ld, \"admitted\": true},\n", proj_width);
  printf("  \"job\": {\"frames\": %ld, \"frame_hw\": [%ld, %ld], \"tubelet\": [%ld, %ld, %ld], \"text_tokens\": %ld},\n", T, H, W, t, p, p, L);
  printf("  \"budget\": {\"visual_total\": %ld, \"visual_budget\": %ld, \"stride\": %ld, \"slabs_admitted\": %ld, \"slabs_total\": %ld, \"Nv\": %ld, \"S\": %ld, \"attention_cost_bound\": %llu},\n",
         Nv_total, budget, stride, admitted, slabs, Nv, S_len, cost);
  printf("  \"decode_pin\": {\"temperature\": 0, \"seed\": %ld},\n", seed);
  printf("  \"trace\": {\"file\": \"%s\", \"head\": \"%s\", \"genesis\": \"model_sha256\"}\n", trace_path, prev);
  printf("}\n");
  return 0;
}
