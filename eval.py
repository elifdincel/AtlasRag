import os, csv, json, argparse
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss, pickle

def hit_at_k(gold_ctx: str, candidates: List[str], k: int = 5) -> int:
    gold = gold_ctx.strip()[:200]
    for c in candidates[:k]:
        if gold[:60].lower() in c.lower() or c.lower() in gold.lower():
            return 1
    return 0

def run_eval(eval_path: str="eval.tsv", index_dir: str="index", k: int=5) -> Dict:
    # load index + meta
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    total = 0
    hits = 0
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            q, gold_ctx = line.rstrip("\n").split("\t")
            qvec = embedder.encode([q], normalize_embeddings=True)[0].astype("float32")
            D, I = index.search(qvec.reshape(1,-1), k)
            cands = [meta[i]["content"] for i in I[0]]
            hits += hit_at_k(gold_ctx, cands, k=k)
            total += 1
    return {"n": total, f"hit@{k}": hits/total if total else 0.0}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_path", default="eval.tsv")
    ap.add_argument("--index_dir", default="index")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    print(json.dumps(run_eval(args.eval_path, args.index_dir, args.k), ensure_ascii=False, indent=2))