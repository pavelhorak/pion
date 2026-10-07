# `cag_legal_demo` — bounded-corpus QA on US Supreme Court opinions

A self-contained demo of `pion-cag-hybrid` on a real legal corpus; reproduces the cascade onboarding flow on a non-SQuAD distribution. Every number below is in `calibration_state.json`, the output of the `pion-serve setup` run that produced it.

## What's in the demo

| File | What |
|---|---|
| `corpus/opinions.txt` | 4 SCOTUS opinions, public domain, 12,327 tokens, 107 paragraphs |
| `calibration_qa.jsonl` | 30 curated Q+A pairs hand-written from the opinion texts |
| `calibration_state.json` | Output of `pion-serve setup`: persisted `(Te, Tm)` + tier + 10-fold CV metrics + per-fold breakdown |
| `sample_queries.jsonl` | 25 additional queries spanning the 4 opinions — operator can try these against the running server |

## Corpus

| Case | Citation | Topic |
|---|---|---|
| Brown v. Board of Education | 347 U.S. 483 (1954) | School desegregation; equal protection |
| Gideon v. Wainwright | 372 U.S. 335 (1963) | Right to counsel in state criminal cases |
| Tinker v. Des Moines | 393 U.S. 503 (1969) | First Amendment in schools (Vietnam armbands) |
| Loving v. Virginia | 388 U.S. 1 (1967) | Anti-miscegenation statutes; strict scrutiny |

Source: [English Wikisource](https://en.wikisource.org/), `{{PD-USGov}}` (federal works are public domain in the U.S.). Wikitext was fetched via the MediaWiki API, stripped of markup, concatenated with case-name headers.

## Calibration result

```
Corpus:       12,327 tokens, 107 paragraphs, sha256[:16]=d08e799f1b68ba6d
Model:        mlx-community/Llama-3.1-8B-Instruct-4bit (4-bit quant)
Calibration:  30 curated QA pairs, cluster=A

Tier fired:   TIER 3 (pure-CAG fallback; no smarter gate fires)
              Selected (Te, Tm) = (99.00, 0.00) — i.e. always keep CAG

In-sample baselines on the 30 calibration queries:
  pure RAG:      F1=0.529  found=0.733  TTFT=1,790 ms
  pure CAG:      F1=0.706  found=0.900  TTFT=1,495 ms  (1.2× faster)

Gated policy (= pure CAG): F1=0.706  found=0.900  fb=0.0%  (1.2× streaming-TTFT)
                           Δ vs RAG: F1 +17.7 pp, found +16.7 pp

Held-out 3-fold CV: F1=0.706  found=0.900  fb=0.0%  (weighted-mean sp 3.77×;
                    folds 0 + 1 picked pure-CAG at 5.26× / 5.50×; fold 2 picked
                    cluster-A center at 0.54× — n=20 train is below the threshold
                    for cluster-stable smart-gate selection on this corpus.)
```

### Reading this honestly

- **Quality wins are large.** Pure CAG dominates pure RAG by **+17.7 pp F1 and +16.7 pp answer-found**. That's the bounded-corpus QA story: the model sees the whole corpus every time, so it answers from the full context instead of from a top-3 retrieval slice.
- **The streaming TTFT win is only 1.2× in-sample.** At 12,327 tokens, the foundation cache is small enough that cold-prefill RAG (which prefills ~300 tokens of retrieved context plus the question) is comparable in latency to the CAG suffix prefill against the cached corpus. A larger corpus should widen the gap, because RAG's prefill stays small while a cold CAG prefill grows with the corpus; this demo does not measure that. The foundation cache could also live in mlx-lm's own prompt-cache file (`save_prompt_cache` / `load_prompt_cache`), which on a single fixed prefix answers faster than a Pion fetch; this demo does not measure that either.
- **Cascade correctly picks pure CAG.** The 3-tier cascade falls through to tier 3 (pure CAG) precisely because it cannot find a smarter `(Te, Tm)` that beats pure CAG on this corpus. Per-fold CV confirms: 2 of 3 folds independently pick pure CAG; fold 2's experimentally smart gate actually regresses (0.54× speedup because fallback overhead dominates). Pure CAG is the right answer here.
- **This corpus is intentionally small.** Run `pion-serve setup` on your own corpus and read its calibration output before expecting a larger TTFT win.

## Deploy

```bash
# Start the pion-cag-hybrid backend with the precomputed calibration
python pion-serve/serve.py --backend pion-cag-hybrid \
    --cag-foundation-corpus examples/cag_legal_demo/corpus/opinions.txt \
    --cag-calibration-state examples/cag_legal_demo/calibration_state.json

# Once the foundation cache is preloaded (~65 s the first time), serve any
# OpenAI-compatible chat request:
curl -sS http://127.0.0.1:8321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pion-cag-hybrid",
    "messages": [
      {"role": "user", "content": "Which constitutional amendment did Tinker rely on for the students free expression claim?"}
    ],
    "max_tokens": 20
  }' | python3 -m json.tool
```

Expected response shape (the `cag_branch` field shows whether the gate kept the CAG answer or fell back to cold RAG; here always `cag_kept` because tier 3 is in effect):

```json
{
  "choices": [{"message": {"role": "assistant", "content": "The First Amendment"}}],
  "cag_branch": "cag_kept",
  "cag_signals": {"entropy": 0.41, "margin": 2.8, "te": 99.0, "tm": 0.0}
}
```

## Re-run the setup yourself

The whole onboarding flow takes ~5 min on Apple Silicon:

```bash
python pion-serve/serve.py setup \
    --corpus examples/cag_legal_demo/corpus/opinions.txt \
    --qa-source examples/cag_legal_demo/calibration_qa.jsonl \
    --cluster A \
    --out examples/cag_legal_demo/calibration_state.json
```

This will regenerate `calibration_state.json` and reprint the operator card. Numbers will be identical to the committed state (greedy decoding, deterministic).

## Replace the corpus with your own

Drop your `.txt` / `.md` files into a directory and point `--corpus` at the directory instead of the single file:

```bash
python pion-serve/serve.py setup --corpus /path/to/your/corpus_dir \
    --qa-count 150 \
    --cluster A \
    --out my_corpus_state.json
```

If you don't have a curated QA set, omit `--qa-source` and the setup generates 150 template-mode questions from the corpus (lower-quality calibration but fully automatic). For production deployments, supply curated QA matching your query distribution.

## Provenance + license

- Opinion text: federal-work public domain (`{{PD-USGov}}` on Wikisource).
- Calibration QA + sample queries: hand-written for this demo, released under the same license as the Pion repository.
- This demo is part of the productization of pion-cag-hybrid.
