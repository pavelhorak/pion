#!/usr/bin/env python3
"""Generate Q&A pairs from Pion documentation for the replay experiment.

Reads markdown files, splits into sections, and generates question-answer
pairs using an LLM backend (ollama, or falls back to template-based
extraction).

Usage:
    # With LLM (better quality):
    python3 pion-serve/generate_qa.py --source doc/ --count 500 --backend ollama --model gemma3:4b

    # Template-based (no LLM needed, instant):
    python3 pion-serve/generate_qa.py --source doc/ --count 500 --no-llm

    # From specific files:
    python3 pion-serve/generate_qa.py --source README.md doc/architecture.md --count 200

Output: JSONL file with {"question": "...", "answer": "...", "source_doc": "...", "variant": 0}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests


def read_markdown_files(sources: list[str]) -> list[dict]:
    """Read markdown files and split into titled sections."""
    sections = []
    for source in sources:
        p = Path(source)
        if p.is_dir():
            files = sorted(p.glob("**/*.md"))
        elif p.is_file():
            files = [p]
        else:
            print(f"Warning: {source} not found, skipping")
            continue

        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"Warning: could not read {f}: {e}")
                continue

            # Skip very short files
            if len(text) < 100:
                continue

            # Split by ## headings
            chunks = re.split(r'\n(?=##\s)', text)
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < 80:
                    continue
                # Extract heading
                heading_match = re.match(r'^##\s+(.+)', chunk)
                heading = heading_match.group(1).strip() if heading_match else ""
                sections.append({
                    "source": str(f),
                    "heading": heading,
                    "content": chunk[:3000],  # cap per section
                })

    return sections


def generate_qa_with_llm(
    sections: list[dict],
    count: int,
    backend_url: str,
    model: str,
    paraphrases: int = 2,
) -> list[dict]:
    """Generate Q&A pairs using an LLM."""
    qa_pairs = []
    # Shuffle sections for variety
    random.shuffle(sections)

    # Calculate how many base questions we need (each gets paraphrases)
    base_count = count // (1 + paraphrases)

    for i, section in enumerate(sections):
        if len(qa_pairs) >= count:
            break

        prompt = (
            f"Based on this documentation section, generate {min(3, max(1, base_count // len(sections) + 1))} "
            f"question-answer pairs. Each question should be something a developer would ask.\n\n"
            f"Documentation:\n{section['content']}\n\n"
            f"Respond ONLY with a JSON array of objects, each with \"question\" and \"answer\" fields. "
            f"Keep answers concise (2-4 sentences). No markdown formatting in the JSON.\n\n"
            f"Example: [{{\"question\": \"How does X work?\", \"answer\": \"X works by...\"}}]"
        )

        try:
            resp = requests.post(
                f"{backend_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.7},
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")

            # Extract JSON array from response
            pairs = _extract_json_array(content)
            for pair in pairs:
                if "question" in pair and "answer" in pair:
                    qa_pairs.append({
                        "question": pair["question"],
                        "answer": pair["answer"],
                        "source_doc": section["source"],
                        "heading": section["heading"],
                        "variant": 0,
                    })

            sys.stdout.write(f"\r  Generated {len(qa_pairs)}/{count} Q&A pairs from {i+1}/{len(sections)} sections...")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n  Warning: LLM call failed for section '{section['heading']}': {e}")
            continue

    print()

    # Generate paraphrased variants
    if paraphrases > 0 and qa_pairs:
        print(f"  Generating {paraphrases} paraphrase(s) per question...")
        originals = list(qa_pairs)  # copy
        for qi, orig in enumerate(originals):
            if len(qa_pairs) >= count:
                break
            for v in range(1, paraphrases + 1):
                if len(qa_pairs) >= count:
                    break
                prompt = (
                    f"Rephrase this question in a different way, keeping the same meaning. "
                    f"Return ONLY the rephrased question, nothing else.\n\n"
                    f"Original: {orig['question']}"
                )
                try:
                    resp = requests.post(
                        f"{backend_url}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"temperature": 0.9},
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    rephrased = resp.json().get("message", {}).get("content", "").strip()
                    if rephrased and len(rephrased) > 10:
                        qa_pairs.append({
                            "question": rephrased,
                            "answer": orig["answer"],
                            "source_doc": orig["source_doc"],
                            "heading": orig["heading"],
                            "variant": v,
                        })
                except Exception:
                    pass

            if (qi + 1) % 10 == 0:
                sys.stdout.write(f"\r  Paraphrased {qi+1}/{len(originals)} questions ({len(qa_pairs)} total)...")
                sys.stdout.flush()
        print()

    return qa_pairs[:count]


def generate_qa_template(sections: list[dict], count: int, paraphrases: int = 2) -> list[dict]:
    """Generate Q&A pairs using template-based extraction (no LLM needed).

    Extracts facts from documentation sections and creates questions using
    common patterns. Faster and deterministic but lower quality than LLM.
    """
    qa_pairs = []
    random.shuffle(sections)

    templates = [
        ("What is {topic}?", "explain"),
        ("How does {topic} work?", "explain"),
        ("What does {topic} do?", "explain"),
        ("How do you use {topic}?", "howto"),
        ("What are the options for {topic}?", "options"),
        ("What is the default {topic}?", "default"),
        ("How do you configure {topic}?", "howto"),
    ]

    paraphrase_templates = [
        "Can you explain {topic}?",
        "Tell me about {topic}.",
        "What's the deal with {topic}?",
        "I need help understanding {topic}.",
        "Describe how {topic} functions.",
    ]

    for section in sections:
        if len(qa_pairs) >= count:
            break

        heading = section["heading"]
        content = section["content"]
        if not heading or len(heading) < 3:
            continue

        # Clean heading for use as topic
        topic = heading.strip("# ").strip()

        # Extract first meaningful paragraph as answer
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]
        # Skip code blocks and tables
        answer_lines = []
        in_code = False
        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if line.startswith("|") or line.startswith("---"):
                continue
            answer_lines.append(line)
            if len(" ".join(answer_lines)) > 300:
                break

        if not answer_lines:
            continue
        answer = " ".join(answer_lines)[:500]

        # Generate base question
        template = random.choice(templates)
        question = template[0].format(topic=topic)

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "source_doc": section["source"],
            "heading": heading,
            "variant": 0,
        })

        # Generate paraphrases
        for v in range(1, min(paraphrases + 1, len(paraphrase_templates))):
            if len(qa_pairs) >= count:
                break
            qa_pairs.append({
                "question": paraphrase_templates[v - 1].format(topic=topic),
                "answer": answer,
                "source_doc": section["source"],
                "heading": heading,
                "variant": v,
            })

    return qa_pairs[:count]


def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM response text (handles markdown fences)."""
    # Strip markdown code fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try to find JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Try line-by-line JSON objects
    results = []
    for line in text.split("\n"):
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from Pion docs")
    parser.add_argument("--source", nargs="+", default=["doc/", "README.md"],
                        help="Source files/directories (default: doc/ README.md)")
    parser.add_argument("--count", type=int, default=500,
                        help="Target number of Q&A pairs (default: 500)")
    parser.add_argument("--output", default="pion-serve/qa_dataset.jsonl",
                        help="Output JSONL file (default: pion-serve/qa_dataset.jsonl)")
    parser.add_argument("--backend-url", default="http://127.0.0.1:11434",
                        help="LLM backend URL (default: ollama localhost)")
    parser.add_argument("--model", default="gemma3:4b",
                        help="LLM model for generation (default: gemma3:4b)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use template-based extraction (no LLM needed)")
    parser.add_argument("--paraphrases", type=int, default=2,
                        help="Number of paraphrased variants per question (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("  Pion Q&A Generator — inference distillation dataset")
    print("=" * 60)

    # Read source docs
    print(f"\n1. Reading source files: {args.source}")
    sections = read_markdown_files(args.source)
    print(f"   Found {len(sections)} sections from {len(set(s['source'] for s in sections))} files")

    if not sections:
        print("ERROR: No documentation sections found")
        sys.exit(1)

    # Generate Q&A pairs
    print(f"\n2. Generating {args.count} Q&A pairs (paraphrases={args.paraphrases})...")
    if args.no_llm:
        print("   Mode: template-based (no LLM)")
        qa_pairs = generate_qa_template(sections, args.count, args.paraphrases)
    else:
        print(f"   Mode: LLM ({args.model} @ {args.backend_url})")
        qa_pairs = generate_qa_with_llm(
            sections, args.count, args.backend_url, args.model, args.paraphrases
        )

    print(f"   Generated {len(qa_pairs)} Q&A pairs")

    if not qa_pairs:
        print("ERROR: No Q&A pairs generated")
        sys.exit(1)

    # Shuffle for replay
    random.shuffle(qa_pairs)

    # Write output
    print(f"\n3. Writing to {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Summary
    sources = set(p["source_doc"] for p in qa_pairs)
    variants = sum(1 for p in qa_pairs if p["variant"] > 0)
    print(f"\n   Total: {len(qa_pairs)} pairs ({len(qa_pairs) - variants} originals, {variants} paraphrases)")
    print(f"   Sources: {len(sources)} files")
    print(f"   Output: {args.output}")
    print()


if __name__ == "__main__":
    main()
