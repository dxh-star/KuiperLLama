
"""导出 GPT-2 词表为 KuiperLLama 可读 tokenizer.json（base/gpt2_bpe.h 解析）。用法: --from-hf <目录/名> 或 --from-tokenizer-json <tokenizer.json> 或 --from-vocab-json <vocab.json>"""
import argparse, json, os
def write_kuiper_json(vocab, merges, added, output):
    json.dump({"added_tokens": added, "model": {"vocab": vocab, "merges": merges}}, open(output, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {output}  vocab={len(vocab)} merges={len(merges)} added={len(added)}")
def from_hf(model_name, output):
    import json, os
    tj = os.path.join(model_name, "tokenizer.json")
    if os.path.exists(tj):
        # 本地 HF 目录：tokenizer.json 就是目标格式，直接读取
        d = json.load(open(tj, encoding="utf-8"))
        write_kuiper_json(d["model"]["vocab"], d["model"]["merges"], d.get("added_tokens", []), output)
        return
    # 兜底：用 HF fast tokenizer
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    vocab = tok.get_vocab()
    merges = [f"{a} {b}" for a, b in tok._tokenizer.model.merges]
    special = tok.special_tokens_map
    seen = {}
    for k, v in special.items():
        if isinstance(v, list):
            for s in v: seen[s] = tok.convert_tokens_to_ids(s)
        elif v is not None: seen[v] = tok.convert_tokens_to_ids(v)
    added = [{"id": i, "content": c} for c, i in seen.items()]
    write_kuiper_json(vocab, merges, added, output)
def from_tokenizer_json(path, output):
    d = json.load(open(path, encoding="utf-8"))
    write_kuiper_json(d["model"]["vocab"], d["model"]["merges"], d.get("added_tokens", []), output)
def from_vocab_json(vocab_path, output):
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    merges_path = os.path.join(os.path.dirname(vocab_path), "merges.txt")
    merges = [l.rstrip("\n") for l in open(merges_path, encoding="utf-8") if l.strip()]
    write_kuiper_json(vocab, merges, [], output)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=str)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-hf", type=str); g.add_argument("--from-tokenizer-json", type=str); g.add_argument("--from-vocab-json", type=str)
    args = ap.parse_args()
    if args.from_hf: from_hf(args.from_hf, args.output)
    elif args.from_tokenizer_json: from_tokenizer_json(args.from_tokenizer_json, args.output)
    else: from_vocab_json(args.from_vocab_json, args.output)
