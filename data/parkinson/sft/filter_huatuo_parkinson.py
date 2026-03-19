# -*- coding: utf-8 -*-
"""
filter_huatuo_parkinson_ascii.py

Filter Parkinson-related QA pairs from Huatuo26M-Lite,
then rank them by cosine similarity to your 50 SR abstracts.

Pipeline:
  Step 1: Load 50 SR abstracts from consensus_ai CSV
  Step 2: Keyword filter on Huatuo26M-Lite (streaming, checkpoint saved)
  Step 3: Embed all texts (local sentence-transformers OR DeepSeek API)
  Step 4: Cosine similarity matrix, keep Top N
  Step 5: Save ranked candidates for DPO chosen/rejected generation

EMBED_MODE options:
  "local"    -- sentence-transformers, free, no internet needed after download
  "deepseek" -- DeepSeek text-embedding-v3 API (NOT deepseek-r1, that is generative)

Install deps:
  pip install datasets numpy tqdm pandas sentence-transformers torch --break-system-packages
  pip install openai --break-system-packages  # only for deepseek mode
"""

import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# =============================================================================
#  CONFIG -- edit only this section
# =============================================================================

EMBED_MODE       = "local"           # "local" or "deepseek"
DEEPSEEK_API_KEY = "none"   # only needed when EMBED_MODE="deepseek"
LOCAL_MODEL      = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SR_CSV           = "consensus ai - Sheet1.csv"
OUTPUT_KEYWORD   = "parkinson_keyword_filtered.jsonl"
OUTPUT_FINAL     = "parkinson_vector_filtered.jsonl"

TOP_N            = 500   # final candidates to keep
EMBED_BATCH      = 16    # batch size for DeepSeek API calls

# =============================================================================

PARKINSON_KEYWORDS = [
    # disease names
    "\u5e15\u91d1\u68ee",           # PD (Chinese)
    "\u5e15\u91d1\u68ee\u75c5",
    "\u5e15\u91d1\u68ee\u7efc\u5408\u5f81",
    "\u5e15\u91d1\u68ee\u6c0f",
    # motor symptoms
    "\u9707\u98a4",                 # tremor
    "\u9759\u6b62\u6027\u9707\u98a4",
    "\u624b\u6296",
    "\u8fd0\u52a8\u8fdf\u7f13",     # bradykinesia
    "\u808c\u5f3a\u76f4",           # rigidity
    "\u51bb\u7ed3\u6b65\u6001",     # freezing of gait
    "\u6162\u6b65\u6001",
    "\u6163\u5f20\u6b65\u6001",
    # non-motor
    "\u5feb\u52a8\u773c\u7761\u7720",  # REM sleep
    "RBD",
    "\u55c5\u89c9\u51cf\u9000",     # hyposmia
    # anatomy / pathology
    "\u9ed1\u8d28",                 # substantia nigra
    "\u7eb9\u72b6\u4f53",           # striatum
    "\u8def\u6613\u4f53",           # Lewy body
    "\u591a\u5df4\u80fa\u80fd\u795e\u7ecf\u5143",
    # diagnostics
    "DaTscan",
    "\u591a\u5df4\u80fa\u8f6c\u8fd0\u4f53",
    "\u7ecf\u9885\u8d85\u58f0",
    # drugs
    "\u5de6\u65cb\u591a\u5df4",     # levodopa
    "\u591a\u5df4\u80fa",           # dopamine
    "\u591a\u5df4\u80fa\u80fd",
    "\u5361\u6bd4\u591a\u5df4",
    "\u666e\u62c9\u514b\u7d22",
    "\u7f57\u5339\u5c3c\u7f57",
    "\u53f8\u6765\u5409\u5170",
    "\u96f7\u6c99\u5409\u5170",
    "\u6069\u4ed6\u5361\u670b",
    "\u91d1\u523a\u70f7\u5b89",
    "\u82ef\u6d77\u7d22",
    # surgery
    "\u8111\u6df1\u90e8\u7535\u523a\u6fc0",  # DBS
    "DBS",
    # English (some Huatuo entries contain English)
    "Parkinson", "levodopa", "dopamine", "bradykinesia", "substantia nigra",
]


# =============================================================================
#  Embedding backend
# =============================================================================

def init_embed_fn():
    """
    Return an embed(texts) -> list[list[float]] function
    based on EMBED_MODE setting.

    Why two modes?
    - local: completely free, works offline after first model download (~400MB).
      Uses sentence-transformers with a multilingual model that handles both
      Chinese and English well.
    - deepseek: uses DeepSeek text-embedding-v3 API. Note: deepseek-r1 is a
      *generative* model and cannot produce embeddings. text-embedding-v3 is
      a separate model but shares the same API key.
    """
    if EMBED_MODE == "local":
        from sentence_transformers import SentenceTransformer
        print("[INFO] Loading local embedding model: " + LOCAL_MODEL)
        print("       First run will download ~400MB, cached afterwards.")
        model = SentenceTransformer(LOCAL_MODEL)
        print("[INFO] Model loaded.\n")

        def embed(texts):
            vecs = model.encode(texts, batch_size=64, show_progress_bar=False)
            return vecs.tolist()
        return embed

    elif EMBED_MODE == "deepseek":
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

        def embed(texts):
            all_vecs = []
            for i in range(0, len(texts), EMBED_BATCH):
                batch = texts[i: i + EMBED_BATCH]
                try:
                    resp = client.embeddings.create(
                        model="text-embedding-v3",
                        input=batch,
                    )
                    all_vecs.extend([item.embedding for item in resp.data])
                except Exception as e:
                    print("[WARN] DeepSeek API error at batch {}: {}".format(i // EMBED_BATCH, e))
                    # zero-vector fallback: these samples will rank last, not affecting Top N
                    all_vecs.extend([[0.0] * 1024] * len(batch))
            return all_vecs
        return embed

    else:
        raise ValueError("EMBED_MODE must be 'local' or 'deepseek', got: " + EMBED_MODE)


# =============================================================================
#  Step 1: Load SR abstracts from CSV as target distribution
# =============================================================================

def load_sr_abstracts():
    """
    Build target-distribution texts from your consensus_ai CSV.

    Each text = Chinese clinical question + first 800 chars of English answer.
    Combining both languages lets the embedding capture cross-lingual semantics,
    so Chinese Huatuo QA pairs that discuss the same clinical topics score high
    even though the SR answers are in English.
    """
    try:
        df = pd.read_csv(SR_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(SR_CSV, encoding="gbk")

    zh_col = [c for c in df.columns if "\u4e2d\u6587" in c][0]  # column containing Chinese question
    en_col = "\u56de\u7b54"  # column name = answer

    abstracts = []
    for _, row in df.iterrows():
        zh = str(row.get(zh_col, "")).strip()
        en = str(row.get(en_col, "")).strip()
        abstracts.append("Clinical question: {}\nAbstract: {}".format(zh, en[:800]))

    print("[INFO] Loaded {} SR abstracts from {}".format(len(abstracts), SR_CSV))
    return abstracts


# =============================================================================
#  Step 2: Keyword filter on Huatuo26M-Lite (streaming + checkpoint)
# =============================================================================

def keyword_filter():
    """
    Stream Huatuo26M-Lite and keep only Parkinson-related QA pairs.

    Why streaming=True?
    The dataset has millions of records. Full download would take tens of GB.
    Streaming reads one record at a time, so memory stays flat regardless of
    how large the dataset is.

    Checkpoint design:
    Filtering the full dataset takes 20-40 minutes. The result is saved to
    OUTPUT_KEYWORD so that if the vector-embedding step fails (e.g. API error),
    you can re-run the script and it skips directly to embedding -- no re-scan.
    """
    print("[Step 2] Streaming Huatuo26M-Lite, keyword filtering...")
    dataset = load_dataset(
        "FreedomIntelligence/Huatuo26M-Lite",
        split="train",
        streaming=True,
    )

    matched = []
    scanned = 0

    for item in tqdm(dataset, desc="  Scanning Huatuo"):
        scanned += 1
        q = item.get("question", "") or ""
        a = item.get("answer",   "") or ""

        if any(kw in q + a for kw in PARKINSON_KEYWORDS):
            matched.append({
                "question":        q,
                "answer":          a,
                # feature_content is the text we embed.
                # Full question (usually short) + first 300 chars of answer.
                "feature_content": "Patient: {}\nDoctor: {}".format(q, a[:300]),
            })

        if scanned % 100_000 == 0:
            print("  Scanned {:,} | Matched {}".format(scanned, len(matched)))

        # 3000 is a generous candidate pool for Parkinson (rare in general medical datasets)
        if len(matched) >= 3000:
            print("  Reached cap of 3000 matches, stopping scan at {:,} records.".format(scanned))
            break

    print("[Step 2] Done. Scanned {:,}, matched {}.".format(scanned, len(matched)))

    with open(OUTPUT_KEYWORD, "w", encoding="utf-8") as f:
        for it in matched:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("[Step 2] Saved to {} (checkpoint for restart)\n".format(OUTPUT_KEYWORD))

    return matched


# =============================================================================
#  Steps 3-4: Embed + cosine similarity ranking
# =============================================================================

def vector_filter(candidates, embed_fn):
    """
    Rank candidates by cosine similarity to SR abstracts, keep Top N.

    Core idea:
      sim_matrix = C (M x D) @ S.T (D x K)  -->  shape (M, K)
    where M = #candidates, K = #SR abstracts, D = embedding dimension.
    Each row is one candidate's similarity to all 50 SR abstracts.
    We then take the mean of each row's top-5 similarities as the score.

    Why top-5 mean instead of all-50 mean?
    Your 50 SRs cover different clinical scenarios (diagnosis, drugs, NMS...).
    A candidate typically matches only a few scenarios strongly. Using all-50
    mean dilutes the score with irrelevant SRs; top-5 mean captures how well
    the candidate aligns with its *most relevant* scenarios.
    """
    print("[Step 3] Embedding SR abstracts...")
    sr_texts = load_sr_abstracts()
    sr_vecs  = embed_fn(sr_texts)
    print("[Step 3] SR embedding done.\n")

    print("[Step 3] Embedding {} candidates...".format(len(candidates)))
    cand_texts = [it["feature_content"] for it in candidates]
    cand_vecs  = []
    batch_size = 256 if EMBED_MODE == "local" else EMBED_BATCH
    for i in tqdm(range(0, len(cand_texts), batch_size), desc="  Embedding candidates"):
        batch = cand_texts[i: i + batch_size]
        cand_vecs.extend(embed_fn(batch))
    print("[Step 3] Candidate embedding done.\n")

    C = np.array(cand_vecs, dtype=np.float32)  # (M, D)
    S = np.array(sr_vecs,   dtype=np.float32)  # (K, D)

    # L2 normalize: dot product of unit vectors = cosine similarity
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-10
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-10

    print("[Step 4] Computing similarity matrix and ranking...")
    sim = C @ S.T  # (M, K)

    k = min(5, S.shape[0])
    top5_idx    = np.argpartition(-sim, k, axis=1)[:, :k]
    top5_scores = np.take_along_axis(sim, top5_idx, axis=1)
    avg_scores  = top5_scores.mean(axis=1)

    sorted_idx = np.argsort(-avg_scores)[:TOP_N]

    print("[Step 4] Done.")
    print("  Top-1 score  : {:.4f}".format(avg_scores[sorted_idx[0]]))
    print("  Top-{} score : {:.4f}".format(TOP_N, avg_scores[sorted_idx[-1]]))
    print("  Median score : {:.4f}\n".format(np.median(avg_scores)))

    results = []
    for rank, idx in enumerate(sorted_idx):
        it = candidates[int(idx)].copy()
        it["similarity_score"] = float(avg_scores[idx])
        it["rank"] = rank + 1
        results.append(it)
    return results


# =============================================================================
#  Step 5: Save and report
# =============================================================================

def save_and_report(results):
    with open(OUTPUT_FINAL, "w", encoding="utf-8") as f:
        for it in results:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    scores = [it["similarity_score"] for it in results]
    print("[Step 5] Saved {} candidates to {}".format(len(results), OUTPUT_FINAL))
    print("  Score stats:")
    print("    Top-50  mean: {:.4f}".format(np.mean(scores[:50])))
    print("    Top-100 mean: {:.4f}".format(np.mean(scores[:100])))
    print("    Top-500 mean: {:.4f}".format(np.mean(scores[:500])))

    print("\nTop-5 preview:")
    for it in results[:5]:
        print("  [#{} {:.4f}] {}...".format(
            it["rank"], it["similarity_score"], it["question"][:60]))

    print("\nNext steps:")
    print("  1. Inspect {} -- check quality of high-score samples".format(OUTPUT_FINAL))
    print("  2. Generate response_chosen for these {} samples via DeepSeek-R1".format(len(results)))
    print("  3. Generate response_rejected via regenerate_rejected.py")
    print("  4. Merge with existing 150 DPO samples -> ~{} total".format(150 + len(results)))


# =============================================================================
#  Main
# =============================================================================

def main():
    print("=" * 60)
    print("Huatuo Parkinson Filter")
    print("Embed mode : {}".format(EMBED_MODE))
    print("Target TOP N: {}".format(TOP_N))
    print("=" * 60 + "\n")

    embed_fn = init_embed_fn()

    # Checkpoint: skip keyword scan if already done
    if os.path.exists(OUTPUT_KEYWORD):
        print("[Step 2] Found checkpoint {}, loading...".format(OUTPUT_KEYWORD))
        candidates = []
        with open(OUTPUT_KEYWORD, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print("[Step 2] Loaded {} candidates.\n".format(len(candidates)))
    else:
        candidates = keyword_filter()

    if not candidates:
        print("[ERROR] No candidates found. Check dataset loading.")
        return

    results = vector_filter(candidates, embed_fn)
    save_and_report(results)


if __name__ == "__main__":
    main()