"""MiniCPM4.1-8B prefill with InfLLM-v2 sparse attention, one layer at a time, on OpenBMB's official kernels
(third_party/infllmv2_cuda_impl, BF16 build).

A lean re-implementation of modeling_minicpm.py's prefill path:
  embed * scale_emb -> 32 x [RMSNorm -> attn -> residual*(scale_depth/sqrt(L)) ;
  RMSNorm -> SwiGLU MLP -> residual*(scale_depth/sqrt(L))] -> RMSNorm -> lm_head(h / (hidden/dim_model_base))
with LongRoPE (fp32 rotary) and GQA 32 Q / 2 KV heads x 128.

InfLLM-v2 sparse attention (kv length >= dense_len = 8192), per layer:
  1. compressed keys: k_c  = mean of 32-token windows, stride 16   (per KV head)
                      k_c2 = mean of 128-token windows, stride 64  (only for the softmax normaliser)
  2. stage 1: every query head scores every causal k_c entry; softmax per head, normalised with an LSE computed on
     the coarser k_c2; probabilities summed over the 16 query heads of a GQA group -> score[kv_head, q, k_c]
  3. block score: max over the k_c entries overlapping each 64-token block; the init block (block 0) and the
     2048-token local window (32 blocks) are forced to +inf
  4. top-k: 64 + 32 = 96 blocks per (kv_head, query token), sorted; future blocks -> -1
  5. block-sparse attention over the selected 64-token blocks (the 16 heads of a group share the selection),
     causal inside the diagonal block.
"""
import json
import math
import os

import torch
import torch.nn.functional as F
from infllm_v2 import infllmv2_attn_stage1, infllmv2_attn_varlen_func, max_pooling_1d_varlen
from safetensors import safe_open

SPARSE = dict(kernel_size=32, kernel_stride=16, init_blocks=1, block_size=64, window_size=2048,
              topk=64, use_nope=False, dense_len=8192)       # MiniCPM4.1 defaults

# LongRoPE long_factor validated by OpenBMB for 128K contexts; used when S > 64K
LONG_FACTOR_128K = [0.9982316082870437, 1.033048153422584, 1.0749920956484724, 1.1255096879436193, 1.1863348602111476, 1.259543828902579, 1.3476188888731149, 1.4535223827776373, 1.5807816745852985, 1.7335856049489526, 1.9168922912975785, 2.1365471404135326, 2.3994084200118646, 2.713475511863602, 3.0880118452194134, 3.533650295140154, 4.062463396503134, 4.687974098908333, 5.425075306704039, 6.289818967956352, 7.29902962722721, 8.6357018163639, 10.210822723989212, 12.053807765671676, 14.193944598909404, 16.65780676784363, 19.463620727694074, 22.628311203524586, 26.150106147261315, 30.02526691405111, 34.23183327975347, 38.73811934094828, 43.502489489729555, 48.47627117965394, 53.61139491762471, 58.857366522037935, 64.16798299215064, 69.51359464319125, 74.86555458220285, 80.21497790341579, 85.55322183307433, 90.89611806932027, 96.26245306514224, 101.68269304046481, 107.18619510219668, 112.82253283014026, 118.63764063163615, 119.88866203644656, 120.9462882391725, 121.837565139014, 122.58663780572562, 123.2147719894291, 123.74049454862576, 124.17980424685767, 124.54641761955492, 124.85202548028222, 125.10654406389756, 125.31835105170659, 125.49450117164764, 125.64091910903052, 125.76256945356558, 125.86360463815589, 125.94749252260765, 126.01712561287873]  # noqa: E501


class Cfg:
    def __init__(self, model_dir):
        c = json.load(open(os.path.join(model_dir, "config.json")))
        self.hidden = c["hidden_size"]
        self.inter = c["intermediate_size"]
        self.n_layers = c["num_hidden_layers"]
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = self.hidden // self.n_heads
        self.eps = c["rms_norm_eps"]
        self.theta = c.get("rope_theta", 10000.0)
        self.scale_emb = c["scale_emb"]
        self.scale_depth = c["scale_depth"]
        self.dim_model_base = c["dim_model_base"]
        rs = c["rope_scaling"]
        self.short_factor = rs["short_factor"]
        self.long_factor = rs["long_factor"]
        self.orig_max = rs["original_max_position_embeddings"]
        self.max_pos = c["max_position_embeddings"]


def rope_cos_sin(cfg, S, device, long_factor=None):
    """MiniCPMLongRoPE: factors = long if S > original_max else short; scaling factor = 1 here."""
    D = cfg.head_dim
    inv = 1.0 / (cfg.theta ** (torch.arange(0, D, 2, device=device).float() / D))
    if S > cfg.orig_max:
        ext = torch.tensor(long_factor or cfg.long_factor, dtype=torch.float32, device=device)
    else:
        ext = torch.tensor(cfg.short_factor, dtype=torch.float32, device=device)
    scale = cfg.max_pos / cfg.orig_max
    mscale = math.sqrt(1 + math.log(scale) / math.log(cfg.orig_max)) if scale > 1 else 1.0
    t = torch.arange(max(S, cfg.max_pos) if S <= cfg.orig_max else S, device=device).float()
    freqs = torch.outer(t, 1.0 / ext) * inv
    emb = torch.cat((freqs, freqs), -1)
    return (emb.cos() * mscale)[:S], (emb.sin() * mscale)[:S]


def apply_rope(x, cos, sin):
    """x [S, H, D] bf16 -> rotated in fp32, back to bf16 (as modeling_minicpm)."""
    xf = x.float()
    x1, x2 = xf[..., : xf.size(-1) // 2], xf[..., xf.size(-1) // 2:]
    rot = torch.cat((-x2, x1), -1)
    return (xf * cos[:, None] + rot * sin[:, None]).to(x.dtype)


def rmsnorm(x, w, eps):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)).to(x.dtype) * w


class Store:
    """safetensors weight loader (by tensor name) of a MiniCPM checkpoint directory"""

    def __init__(self, model_dir, device):
        self.dir, self.device = model_dir, device
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]

    def load(self, keys):
        by = {}
        for k in keys:
            by.setdefault(self.map[k], []).append(k)
        out = {}
        for fn, ks in by.items():
            with safe_open(os.path.join(self.dir, fn), framework="pt", device=str(self.device)) as f:
                for k in ks:
                    out[k] = f.get_tensor(k)
        return out


# ------------------------------------------------------------------ InfLLM-v2 pieces
def compress_k(k, size, stride):
    """k [S, Hkv, D] -> mean of `size`-token windows every `stride` tokens: [n, Hkv, D]."""
    S = k.size(0)
    n = (S - size) // stride + 1 if S >= size else 0
    if n <= 0:
        return k.new_zeros(0, *k.shape[1:])
    w = k.unfold(0, size, stride)                 # [n, Hkv, D, size]
    return w.float().mean(-1).to(k.dtype)


def select_topk(q, k, sp=SPARSE, q_offset=0):
    """InfLLM-v2 block selection for queries q [Lq, H, D] (positions q_offset..q_offset+Lq-1)
    against keys k [q_offset+Lq, Hkv, D] (all keys up to the last query).
    Returns topk_idx [Hkv, Lq, topk] int32 (block ids, -1 = none)."""
    Lq = q.size(0)
    Sk = k.size(0)
    assert Sk == q_offset + Lq
    kc = compress_k(k, sp["kernel_size"], sp["kernel_stride"])
    kc2 = compress_k(k, sp["kernel_size"] * 4, sp["kernel_stride"] * 4)
    dev = q.device
    cu_q = torch.tensor([0, Lq], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kc.size(0)], dtype=torch.int32, device=dev)
    cu_k2 = torch.tensor([0, kc2.size(0)], dtype=torch.int32, device=dev)
    score = infllmv2_attn_stage1(q.contiguous(), kc.contiguous(), kc2.contiguous(), cu_seqlens_q=cu_q,
                                 cu_seqlens_k=cu_k, cu_seqlens_v=cu_k2, max_seqlen_q=Lq,
                                 max_seqlen_k=kc.size(0), causal=True)
    score = score[:, :Lq, :]
    local_blocks = sp["window_size"] // sp["block_size"]
    cache = torch.tensor([q_offset], dtype=torch.int32, device=dev)
    bs = max_pooling_1d_varlen(score.contiguous(), cu_q, cu_k, cache, Lq, kc.size(0),
                               local_blocks=local_blocks, init_blocks=sp["init_blocks"],
                               block_size=sp["block_size"], stride=sp["kernel_stride"])
    topk = min(sp["topk"] + local_blocks, bs.size(-1))
    idx = bs.topk(topk, dim=-1).indices.sort(-1).values
    q_blk = (torch.arange(Lq, device=dev) + q_offset) // sp["block_size"]
    idx[idx > q_blk[None, :, None]] = -1
    return idx.to(torch.int32)


def sparse_attention(q, k, v, topk_idx, q_offset=0):
    """Block-sparse attention of q [Lq,H,D] over k/v [q_offset+Lq,Hkv,D] (bottom-right causal)."""
    Lq, Sk = q.size(0), k.size(0)
    dev = q.device
    cu_q = torch.tensor([0, Lq], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, Sk], dtype=torch.int32, device=dev)
    return infllmv2_attn_varlen_func(q.contiguous(), k.contiguous(), v.contiguous(), cu_q, cu_k, Lq, Sk,
                                     causal=True, topk_idx=topk_idx)


def dense_attention(q, k, v, q_offset=0):
    """Reference dense causal attention (GQA expanded, fp32 math) for the short-sequence path."""
    rep = q.size(1) // k.size(1)
    kk = k.repeat_interleave(rep, 1).transpose(0, 1)
    vv = v.repeat_interleave(rep, 1).transpose(0, 1)
    qq = q.transpose(0, 1)
    Lq, Sk = q.size(0), k.size(0)
    out = torch.empty_like(qq)
    for a in range(0, Lq, 1024):                       # bottom-right causal, query chunks
        b = min(Lq, a + 1024)
        s = torch.matmul(qq[:, a:b].float(), kk.float().transpose(1, 2)) * q.size(-1) ** -0.5
        pos = torch.arange(a, b, device=q.device)[:, None] + (Sk - Lq)
        s = s.masked_fill(torch.arange(Sk, device=q.device)[None, :] > pos, float("-inf"))
        out[:, a:b] = torch.matmul(s.softmax(-1), vv.float()).to(q.dtype)
    return out.transpose(0, 1)


# ------------------------------------------------------------------ layer
class Layer:
    def __init__(self, cfg, i, store, W=None):
        self.cfg, self.i = cfg, i
        p = f"model.layers.{i}."
        if W is None:
            W = store.load([k for k in store.map if k.startswith(p)])
            W = {k[len(p):]: v for k, v in W.items()}
        self.W = W

    def qkv(self, x, cos, sin):
        c, W = self.cfg, self.W
        q = F.linear(x, W["self_attn.q_proj.weight"]).view(-1, c.n_heads, c.head_dim)
        k = F.linear(x, W["self_attn.k_proj.weight"]).view(-1, c.n_kv, c.head_dim)
        v = F.linear(x, W["self_attn.v_proj.weight"]).view(-1, c.n_kv, c.head_dim)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    @torch.inference_mode()
    def forward(self, h, cos, sin, capture=None, tchunk=16384, sparse=True):
        """h [S, hidden] bf16 (updated in place). capture: dict for 'topk' / 'x_attn'."""
        c, W = self.cfg, self.W
        S = h.size(0)
        res_scale = c.scale_depth / math.sqrt(c.n_layers)
        x = rmsnorm(h, W["input_layernorm.weight"], c.eps)
        if capture is not None and capture.get("want_x"):
            capture["x_attn"] = x.clone()
        q, k, v = self.qkv(x, cos, sin)
        del x
        if sparse and S >= SPARSE["dense_len"]:
            idx = select_topk(q, k)
            if capture is not None:
                capture["topk"] = idx
            o = sparse_attention(q, k, v, idx)
        else:
            o = dense_attention(q, k, v)
        del q, k, v
        o = F.linear(o.reshape(S, -1), W["self_attn.o_proj.weight"])
        h.add_(o * res_scale)
        del o
        for a in range(0, S, tchunk):
            b = min(S, a + tchunk)
            x = rmsnorm(h[a:b], W["post_attention_layernorm.weight"], c.eps)
            g = F.linear(x, W["mlp.gate_proj.weight"])
            u = F.linear(x, W["mlp.up_proj.weight"])
            y = F.linear(F.silu(g) * u, W["mlp.down_proj.weight"])
            h[a:b].add_(y * res_scale)
        return h


class EmbedHead:
    def __init__(self, cfg, store):
        self.cfg = cfg
        W = store.load(["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"])
        self.emb, self.norm, self.head = W["model.embed_tokens.weight"], W["model.norm.weight"], W["lm_head.weight"]

    def embed(self, ids):
        return F.embedding(ids, self.emb) * self.cfg.scale_emb

    @torch.inference_mode()
    def logits(self, h):
        c = self.cfg
        x = rmsnorm(h, self.norm, c.eps)
        return F.linear(x / (c.hidden / c.dim_model_base), self.head).float()

    @torch.inference_mode()
    def nll(self, h, ids, chunk=2048):
        tot, hit, n = 0.0, 0, 0
        for a in range(0, h.size(0) - 1, chunk):
            b = min(h.size(0) - 1, a + chunk)
            lg = self.logits(h[a:b])
            tgt = ids[a + 1:b + 1]
            tot += F.cross_entropy(lg, tgt, reduction="sum").item()
            hit += (lg.argmax(-1) == tgt).sum().item()
            n += b - a
        return tot / n, hit / n
