# Eval QA Review Guide

Use this checklist to standardize manual review of `eval/eval_set.json`.

## Goal

Create a robust eval set that reflects real research questions, avoids retrieval overfitting, and keeps every QA pair grounded in a single chunk.

## Pass/Fail Checklist (per QA item)

Must pass all:

- Answerable from the single ground-truth chunk.
- Question is specific enough that this chunk is clearly the best answer.
- Question is phrased differently from the chunk (no copy/paste or near-duplicate wording).
- Reference answer is 1–3 sentences, factual, and supported by the chunk.
- Ground truth fields are correct:
  - `chunk_uids` contains the correct chunk.
  - `papers` contains the correct paper ID.
  - `pages` contains the correct page number(s).
- Difficulty label is either `factual` or `synthesis` and matches the reasoning required.

## Rejection Reasons

Reject or rewrite if any of the following apply:

- Requires context outside the chunk (paper-wide, prior sections, tables/figures not in text).
- Overly generic or broad question with many possible answers.
- Direct keyword mirroring of the chunk without paraphrasing.
- Answer includes citations or speculative text not supported by the chunk.
- Incorrect or missing ground-truth fields.

## Quality Targets (set-level)

- Mix of question types: ~50% factual, ~50% synthesis.
- Coverage across topics (methods, datasets, metrics, limitations).
- Avoid clustering too many questions from the same paper/page.
- Include a few “near‑miss” questions that are still answerable but less obvious.

## Quick Triage Workflow

1. Scan question + chunk text. If it reads like a paraphrase of the chunk, rewrite.
2. Verify the answer is fully contained in the chunk. If not, reject.
3. Check ground-truth IDs and pages.
4. Ensure difficulty tag matches reasoning.

## Minimal Edits Allowed

- Rephrase the question to reduce lexical overlap.
- Trim or reword the reference answer to align with the chunk.
- Update `difficulty` to match the reasoning.
- Correct ground-truth fields.

## Optional Metadata (if you want to add later)

- `expected_topics`: Short topic tags for clustering or error analysis.
- `review_notes`: Reviewer notes for edge cases.
