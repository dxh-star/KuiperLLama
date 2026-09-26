
"""导出 Qwen2.5 HF checkpoint 为 KuiperLLama 的 .bin。--version 1=FP32, 3=int8(Q8_0, 仅量化 matmul 权重; embedding/rmsnorm/bias 保持 fp32)。布局见 qwen2.cpp::create_param_layers/quant_layers。"""
import argparse, json, struct, os
import numpy as np

def load_model(dirp):
    cfg = json.load(open(os.path.join(dirp, "config.json")))
    dim, hidden, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    vocab, tied = cfg["vocab_size"], cfg.get("tie_word_embeddings", True)
    kv_dim = dim * kv_heads // heads
    from safetensors.torch import load_file
    sd = load_file(os.path.join(dirp, "model.safetensors"))
    f32 = lambda k: sd[k].float().cpu().numpy().astype(np.float32).reshape(-1)
    return dict(dim=dim, hidden=hidden, L=L, heads=heads, kv_heads=kv_heads, vocab=vocab, kv_dim=kv_dim, tied=tied), f32

def quantize_q80(w, gs):
    n = w.size; g = w.reshape(-1, gs)
    amax = np.abs(g).max(axis=1, keepdims=True)
    scale = np.where(amax == 0, 1e-8, amax / 127.0)
    return np.round(g / scale).astype(np.int8), scale.reshape(-1).astype(np.float32)

def export_fp32(dirp, outpath, seq_len):
    info, f32 = load_model(dirp)
    with open(outpath, "wb") as out:
        out.write(struct.pack("iiiiiii", info["dim"], info["hidden"], info["L"], info["heads"],
                              info["kv_heads"], info["vocab"] if info["tied"] else -info["vocab"], seq_len))
        ser = lambda a: out.write(a.astype(np.float32).tobytes())
        ser(f32("model.embed_tokens.weight"))
        for i in range(info["L"]):
            p = f"model.layers.{i}"
            ser(f32(f"{p}.self_attn.q_proj.weight")); ser(f32(f"{p}.self_attn.q_proj.bias"))
            ser(f32(f"{p}.self_attn.k_proj.weight")); ser(f32(f"{p}.self_attn.k_proj.bias"))
            ser(f32(f"{p}.self_attn.v_proj.weight")); ser(f32(f"{p}.self_attn.v_proj.bias"))
            ser(f32(f"{p}.self_attn.o_proj.weight"))
            ser(f32(f"{p}.input_layernorm.weight")); ser(f32(f"{p}.post_attention_layernorm.weight"))
            ser(f32(f"{p}.mlp.gate_proj.weight")); ser(f32(f"{p}.mlp.up_proj.weight")); ser(f32(f"{p}.mlp.down_proj.weight"))
        ser(f32("model.norm.weight"))
    return os.path.getsize(outpath)

def export_int8(dirp, outpath, seq_len, group_size=64):
    info, f32 = load_model(dirp)
    with open(outpath, "wb") as out:
        out.write(struct.pack("iiiiiii", info["dim"], info["hidden"], info["L"], info["heads"],
                              info["kv_heads"], info["vocab"] if info["tied"] else -info["vocab"], seq_len))
        out.write(struct.pack("i", group_size))
        serf = lambda a: out.write(a.astype(np.float32).tobytes())
        seri = lambda a: out.write(a.astype(np.int8).tobytes())
        def quant(name):
            w = f32(name); q, s = quantize_q80(w, group_size); seri(q); serf(s)
        serf(f32("model.embed_tokens.weight"))
        for i in range(info["L"]):
            p = f"model.layers.{i}"
            quant(f"{p}.self_attn.q_proj.weight"); serf(f32(f"{p}.self_attn.q_proj.bias"))
            quant(f"{p}.self_attn.k_proj.weight"); serf(f32(f"{p}.self_attn.k_proj.bias"))
            quant(f"{p}.self_attn.v_proj.weight"); serf(f32(f"{p}.self_attn.v_proj.bias"))
            quant(f"{p}.self_attn.o_proj.weight")
            serf(f32(f"{p}.input_layernorm.weight")); serf(f32(f"{p}.post_attention_layernorm.weight"))
            quant(f"{p}.mlp.gate_proj.weight"); quant(f"{p}.mlp.up_proj.weight"); quant(f"{p}.mlp.down_proj.weight")
        serf(f32("model.norm.weight"))
    return os.path.getsize(outpath)

def fmt(b):
    for u in ("B","KB","MB","GB"):
        if b < 1024: return f"{b:>8.2f} {u}"
        b /= 1024
    return f"{b:>8.2f} TB"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=str); ap.add_argument("--hf", default="models/qwen2_hf")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--version", type=int, default=3, choices=[1, 3]); ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()
    info, _ = load_model(args.hf)
    print(f"dim={info['dim']} hidden={info['hidden']} L={info['L']} heads={info['heads']} kv_heads={info['kv_heads']} vocab={info['vocab']} tied={info['tied']}")
    fp32_path = args.out if args.version == 1 else args.out.replace(".bin", "_fp32.bin")
    fp32_sz = export_fp32(args.hf, fp32_path, args.seq_len)
    print(f"FP32 -> {fp32_path} ({fmt(fp32_sz)})")
    int8_sz = None
    if args.version == 3:
        int8_sz = export_int8(args.hf, args.out, args.seq_len, args.group_size)
        print(f"int8 -> {args.out} ({fmt(int8_sz)})")
    else:
        print("提示: 本次只导出了 FP32（--version 1），无 int8 数据可对比。")
    if int8_sz:
        print(f"\n{'='*60}\n Qwen 量化前后对比（单 token 解码, batch=1, seq_len=%d）\n{'='*60}" % args.seq_len)
        print(f"{'模型文件大小':<20s} {fmt(fp32_sz):>12s} {fmt(int8_sz):>12s} {fp32_sz/int8_sz:>8.2f}x")
