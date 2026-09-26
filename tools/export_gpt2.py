
"""导出 GPT-2（HF）checkpoint 为 KuiperLLama 的 .bin。--version 1=FP32, 3=int8(Q8_0, 仅量化 matmul 权重, bias/embedding/ln 保持 fp32)。布局见 gpt2.cpp 的 create_param_layers/quant_layers。"""
import argparse, os, struct
import numpy as np
import torch

def load_hf_gpt2(model_name_or_path):
    from transformers import GPT2LMHeadModel, PretrainedConfig
    config = PretrainedConfig.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    sd = model.state_dict()
    info = {"dim": config.n_embd, "hidden_dim": 4 * config.n_embd, "n_layers": config.n_layer,
            "n_heads": config.n_head, "n_kv_heads": config.n_head,
            "vocab_size": config.vocab_size, "max_seq_len": config.n_positions}
    weights = {"_tied": True}
    weights["wte"] = sd["transformer.wte.weight"].float()
    weights["wpe"] = sd["transformer.wpe.weight"].float()
    for i in range(info["n_layers"]):
        p = f"transformer.h.{i}"; d = info["dim"]
        weights[f"l{i}.ln1.gamma"] = sd[f"{p}.ln_1.weight"].float(); weights[f"l{i}.ln1.beta"] = sd[f"{p}.ln_1.bias"].float()
        ca_w = sd[f"{p}.attn.c_attn.weight"].float().t(); ca_b = sd[f"{p}.attn.c_attn.bias"].float()
        wq_w, wk_w, wv_w = ca_w.split(d, dim=0); wq_b, wk_b, wv_b = ca_b.split(d, dim=0)
        weights[f"l{i}.wq.w"], weights[f"l{i}.wq.b"] = wq_w, wq_b
        weights[f"l{i}.wk.w"], weights[f"l{i}.wk.b"] = wk_w, wk_b
        weights[f"l{i}.wv.w"], weights[f"l{i}.wv.b"] = wv_w, wv_b
        weights[f"l{i}.wo.w"] = sd[f"{p}.attn.c_proj.weight"].float().t(); weights[f"l{i}.wo.b"] = sd[f"{p}.attn.c_proj.bias"].float()
        weights[f"l{i}.ln2.gamma"] = sd[f"{p}.ln_2.weight"].float(); weights[f"l{i}.ln2.beta"] = sd[f"{p}.ln_2.bias"].float()
        weights[f"l{i}.w1.w"] = sd[f"{p}.mlp.c_fc.weight"].float().t(); weights[f"l{i}.w1.b"] = sd[f"{p}.mlp.c_fc.bias"].float()
        weights[f"l{i}.w2.w"] = sd[f"{p}.mlp.c_proj.weight"].float().t(); weights[f"l{i}.w2.b"] = sd[f"{p}.mlp.c_proj.bias"].float()
    weights["lnf.gamma"] = sd["transformer.ln_f.weight"].float(); weights["lnf.beta"] = sd["transformer.ln_f.bias"].float()
    weights["lm_head.weight"] = sd["lm_head.weight"].float()
    weights["_tied"] = torch.allclose(weights["wte"], weights["lm_head.weight"])
    return weights, info

def serialize_fp32(f, t): d=t.detach().cpu().view(-1).to(torch.float32).numpy(); f.write(struct.pack(f"{len(d)}f", *d))
def serialize_int8(f, t): d=t.detach().cpu().view(-1).numpy().astype(np.int8); f.write(struct.pack(f"{len(d)}b", *d))
def quantize_q80(w, gs):
    w = w.float().reshape(-1, gs); wmax = torch.abs(w).max(dim=1).values
    scale = torch.where(wmax == 0, torch.full_like(wmax, 1e-8), wmax / 127.0)
    q = torch.round(w / scale[:, None]).to(torch.int8)
    err = (w - q.float() * scale[:, None]).abs().max().item()
    return q, scale, err

def export_gpt2_fp32(weights, info, filepath):
    d, L = info["dim"], info["n_layers"]; vs = info["vocab_size"] if weights["_tied"] else -info["vocab_size"]
    out = open(filepath, "wb")
    out.write(struct.pack("iiiiiii", d, info["hidden_dim"], L, info["n_heads"], info["n_kv_heads"], vs, info["max_seq_len"]))
    serialize_fp32(out, weights["wte"]); serialize_fp32(out, weights["wpe"])
    for i in range(L):
        serialize_fp32(out, weights[f"l{i}.ln1.gamma"]); serialize_fp32(out, weights[f"l{i}.ln1.beta"])
        for proj in ("wq","wk","wv","wo"): serialize_fp32(out, weights[f"l{i}.{proj}.w"]); serialize_fp32(out, weights[f"l{i}.{proj}.b"])
        serialize_fp32(out, weights[f"l{i}.ln2.gamma"]); serialize_fp32(out, weights[f"l{i}.ln2.beta"])
        for proj in ("w1","w2"): serialize_fp32(out, weights[f"l{i}.{proj}.w"]); serialize_fp32(out, weights[f"l{i}.{proj}.b"])
    serialize_fp32(out, weights["lnf.gamma"]); serialize_fp32(out, weights["lnf.beta"])
    if not weights["_tied"]: serialize_fp32(out, weights["lm_head.weight"])
    out.close(); return os.path.getsize(filepath)

def export_gpt2_int8(weights, info, filepath, group_size=64):
    d, L = info["dim"], info["n_layers"]; vs = -info["vocab_size"]
    out = open(filepath, "wb")
    out.write(struct.pack("iiiiiii", d, info["hidden_dim"], L, info["n_heads"], info["n_kv_heads"], vs, info["max_seq_len"]))
    out.write(struct.pack("i", group_size))
    serialize_fp32(out, weights["wte"]); serialize_fp32(out, weights["wpe"])
    total_err = []
    for i in range(L):
        serialize_fp32(out, weights[f"l{i}.ln1.gamma"]); serialize_fp32(out, weights[f"l{i}.ln1.beta"])
        for proj in ("wq","wk","wv","wo"):
            q,s,err = quantize_q80(weights[f"l{i}.{proj}.w"], group_size)
            serialize_int8(out, q); serialize_fp32(out, s); total_err.append((f"l{i}.{proj}.w", err))
            serialize_fp32(out, weights[f"l{i}.{proj}.b"])
        serialize_fp32(out, weights[f"l{i}.ln2.gamma"]); serialize_fp32(out, weights[f"l{i}.ln2.beta"])
        for proj in ("w1","w2"):
            q,s,err = quantize_q80(weights[f"l{i}.{proj}.w"], group_size)
            serialize_int8(out, q); serialize_fp32(out, s); total_err.append((f"l{i}.{proj}.w", err))
            serialize_fp32(out, weights[f"l{i}.{proj}.b"])
    serialize_fp32(out, weights["lnf.gamma"]); serialize_fp32(out, weights["lnf.beta"])
    q,s,err = quantize_q80(weights["lm_head.weight"], group_size); serialize_int8(out, q); serialize_fp32(out, s); total_err.append(("lm_head.w", err))
    out.close(); total_err.sort(key=lambda x:x[1], reverse=True)
    print(f"  最大量化误差: {total_err[0][1]:.6f} ({total_err[0][0]})")
    return os.path.getsize(filepath)

def fmt(b):
    for u in ("B","KB","MB","GB"):
        if b < 1024: return f"{b:>8.2f} {u}"
        b /= 1024
    return f"{b:>8.2f} TB"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("filepath", type=str)
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--version", type=int, default=3, choices=[1, 3])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()
    weights, info = load_hf_gpt2(args.model)
    fp32_path = args.filepath if args.version == 1 else args.filepath.replace(".bin", "_fp32.bin")
    fp32_sz = export_gpt2_fp32(weights, info, fp32_path)
    print(f"FP32 -> {fp32_path} ({fmt(fp32_sz)})")
    int8_sz = None
    if args.version == 3:
        int8_sz = export_gpt2_int8(weights, info, args.filepath, args.group_size)
        print(f"int8 -> {args.filepath} ({fmt(int8_sz)})")
    if args.no_compare: raise SystemExit(0)
    if int8_sz is None:
        print("提示: 本次只导出了 FP32（--version 1），无 int8 数据可对比。"); raise SystemExit(0)
    print(f"\n{'='*60}\n 量化前后对比（单 token 解码, batch=1）\n{'='*60}")
    print(f"{'模型文件大小':<20s} {fmt(fp32_sz):>12s} {fmt(int8_sz):>12s} {fp32_sz/int8_sz:>8.2f}x")
    print(f"{'总参数量':<20s} {sum(v.numel() for v in weights.values() if isinstance(v, torch.Tensor)):>12,}")
