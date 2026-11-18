# DUA Guard and AI-DUA-Bench

**Compliance-aware AI for research Data Use Agreements.**
Generative AI and Large Language Models (LLM) final project — a
two-person group project by **Abishek Pokhrel and Himal Aryal**,
with contributions shared equally across system design, benchmark
construction, evaluation, and report/demo preparation.

> ⚠️ **Disclaimer — read first.** DUA Guard gives an evidence-based
> *interpretation* of representative clause sets. It is **not legal
> advice** and **not authoritative legal or institutional compliance
> advice**. The bundled clause sets are paraphrased from common public
> Data Use Agreement patterns for classroom evaluation; they are **not**
> verbatim legal documents and **not** the actual agreements of ICPSR,
> PhysioNet, or UK Biobank. Always check the authoritative agreement and
> consult your institution before acting on restricted data.

## The problem

Researchers increasingly use generative AI (cloud chatbots, local
models, coding agents) in workflows involving datasets governed by
restrictive Data Use Agreements (DUAs). A DUA may forbid sharing data
with third parties, require approval for new uses, or limit analysis to
approved systems — but researchers often cannot tell whether a specific
AI-assisted action complies. Unverified LLM advice can hallucinate
permission where none exists, and a false approval is the most
consequential error: it green-lights a violation.

DUA Guard addresses this by grounding every decision in retrieved
agreement clauses, verifying citations verbatim, and abstaining —
*Insufficient Information / Requires Human Review* — when the evidence
is weak, conflicting, or the request omits a decision-relevant fact.

## Reduced scope (per instructor feedback)

The October 2026 proposal planned 10–20 agreements, 5–10 AI policies,
200 scenarios, and several commercial and open-source models. Per
instructor feedback, the built system is deliberately smaller:

| | Proposed | Built |
|---|---|---|
| Dataset agreements | 10–20 | **4** (ICPSR-style, PhysioNet-style, UK Biobank-style, public-use control) |
| AI policies | 5–10 | **3** (cloud LLM, local model, AI coding agent) |
| Benchmark scenarios | 200 | **40** (30–50 as recommended) |
| Labels | 3 | **4** (adds Insufficient Information / Requires Human Review) |
| Models compared | several commercial + open-source | **1 free local model path** (FLAN-T5) + transparent fallback; 3 pipeline settings compared instead |

The full mapping from each instructor point to what was built is in
`report/FINAL_REPORT.md` (Section 2).

## Quickstart (no API key needed)

```bash
# 1. Install (Python 3.10+ recommended; use a virtual environment)
pip install -r requirements.txt

# 2. Run the evaluation (writes results/evaluation_results.json etc.)
python evaluate.py

# 3. Validate the benchmark data
python tests/validate_benchmark.py

# 4. Launch the app
streamlit run app.py
```

Everything runs fully offline with the required dependencies only. No
data leaves your machine, no account is needed, and there is no cost.

## What you get

- **Working app** (`app.py`): check any proposed action against a dataset
  agreement + AI policy; browse all 40 benchmark scenarios with ground
  truth, evidence, and uncertainty; view saved evaluation results; read
  the clause sets.
- **Source code** (`dua_guard/`): clean, commented modules — see
  `HOW_IT_WORKS.md` for a plain-English tour.
- **Benchmark** (`data/benchmark.json`, `data/benchmark.csv`): 40
  synthetic scenarios, 10 of each label, each with evidence clause IDs,
  rationale, and documented uncertainty. No PHI anywhere.
- **Evaluation** (`evaluate.py` + `results/`): real run results for
  three settings (below). Re-run anytime to reproduce them.
- **Final report** (`report/FINAL_REPORT.md`) and **demo script**
  (`presentation/DEMO_SCRIPT.md`).

## How the pipeline works

```
Proposed action
  -> Clause retrieval (RAG): top-k clauses from the selected dataset
     agreement + AI policy (TF-IDF cosine by default; embeddings optional)
  -> Classification: local LLM (FLAN-T5) when available, otherwise the
     transparent evidence-grounded scorer (retrieval + generic cues)
  -> Citation verification: quotes must match a retrieved clause verbatim;
     Allowed needs a verified permission citation
  -> Conservative abstention (Setting C only): weak/conflicting evidence
     or missing context -> Insufficient Information / Requires Human Review
  -> Explanation + next steps (templates filled only with verified quotes)
```

## GenAI components (6 distinct components)

Each component below does a different job; they are not three prompts
for the same task. See the report (Section 4) for the full justification
against the assignment requirements.

1. **Document intelligence / information extraction** (`data_loader.py`):
   agreements and policies become structured clause records (id, title,
   text, stance, topics); user-pasted agreement text is split into
   clauses by `extract_clauses_from_text`.
2. **Embeddings / semantic similarity + RAG retrieval** (`retrieval.py`):
   clauses are indexed and the most relevant ones retrieved per query.
   Default backend is NumPy TF-IDF (offline, deterministic); dense
   embeddings (`all-MiniLM-L6-v2`) and scikit-learn TF-IDF are optional
   backends selected via `DUA_RETRIEVAL_BACKEND`. The backend that
   actually ran is always recorded.
3. **LLM-based classification with zero/few-shot prompting and structured
   output** (`llm_client.py`, `classify.py`): FLAN-T5 is prompted with
   retrieved clauses and must answer in strict JSON
   (`{"label", "quote", "reason"}`); unparseable output falls back
   per-case and the fallback is counted.
4. **Multi-stage pipeline with reasoning** (`classify.py`): retrieval,
   scoring, verification, and abstention are separate stages with
   inspectable intermediate outputs (retrieved clauses, scores, matched
   cues, override reasons).
5. **Output verification / hallucination mitigation** (`verify.py`):
   every displayed quote is checked verbatim against a retrieved clause;
   unverified quotes are dropped, never repaired or paraphrased.
6. **Explanation and recommendation generation** (`explain.py`):
   template explanations and next steps filled only with verified
   citations, so the explanation cannot invent clause language.

## Classification engine: honest status

The evaluation results shipped in `results/` were produced by the
**evidence-grounded fallback engine**, NOT by the FLAN-T5 model. In the
build environment, `transformers`/`torch` were not installed, so the
local LLM could not load; `evaluate.py` recorded this in
`evaluation_results.json` (`engine_info`) instead of silently claiming
LLM results. The fallback scorer is fully transparent: retrieval
similarity over clause stances plus generic, human-readable cue patterns
(see `dua_guard/classify.py`). It never reads benchmark labels or
scenario IDs.

To try the LLM path on a machine with enough memory (~2 GB free):

```bash
pip install transformers torch  # CPU torch build is fine
DUA_CHECK_LLM=1 python evaluate.py
```

If the model loads, Settings B and C use it (with the same verification
and abstention rules); if it fails, the fallback runs and the reason is
recorded. To try embedding retrieval:

```bash
pip install sentence-transformers
DUA_RETRIEVAL_BACKEND=embeddings python evaluate.py
```

## Evaluation results (real run, 40 scenarios)

Run with `python evaluate.py`. Definitions: **false approval rate** =
true Prohibited predicted as Allowed / all true Prohibited (primary
safety metric); **citation accuracy** = predictions with ≥1 citation
where ≥1 cited clause is in the scenario's evidence set; **committed
accuracy** = accuracy over non-abstention predictions only.

| Setting | Accuracy | False approval rate | Unsafe rate | Citation accuracy | Abstention rate | Committed accuracy | Macro-F1 |
|---|---|---|---|---|---|---|---|
| A: No-RAG (question only) | 0.925 (37/40) | 0.000 (0/10) | 0.000 | n/a (no citations) | 0.250 | 0.900 | 0.924 |
| B: LLM + RAG | 0.925 (37/40) | 0.000 (0/10) | 0.000 | 0.650 (26/40) | 0.175 | 0.909 | 0.923 |
| C: Full DUA Guard | 0.925 (37/40) | 0.000 (0/10) | 0.000 | 0.675 (27/40) | 0.225 | 0.935 | 0.924 |

Read this honestly: headline accuracy is identical across settings, so
accuracy alone would hide the real differences. Setting A has no
citations at all and makes one unsafe approval of an approval-gated
action (S20). Setting B never abstains appropriately on underspecified
cases (3 of 10 Insufficient cases committed as Prohibited). Setting C
has the best citation accuracy and the best committed accuracy, and it
is the only setting that both cites evidence and abstains — at the cost
of one over-abstention on a permitted case (S08). With only 10
Prohibited cases, a 0.000 false approval rate is encouraging but not
proof of safety in general; see the report's limitations section.

Full per-label, per-category, confusion-matrix, latency, and
failure-analysis results are in `results/evaluation_results.json`,
`results/predictions.csv`, and `results/confusion_matrix.csv`, and are
discussed in the report (Section 6).

## Project structure

```
dua-guard/
  app.py                  Streamlit application (4 views)
  evaluate.py             Evaluation harness (Settings A/B/C, all metrics)
  requirements.txt        Required + optional dependencies
  README.md               This file
  HOW_IT_WORKS.md         Plain-English code tour (for defending the code)
  dua_guard/
    __init__.py           Labels + disclaimer constants
    data_loader.py        Load agreements/policies/benchmark; custom-DUA extraction
    retrieval.py          TF-IDF / embedding retrievers (RAG retrieval)
    llm_client.py         Free local FLAN-T5 client + honest availability reporting
    classify.py           Cue + retrieval scoring; Settings A/B/C; abstention rules
    verify.py             Verbatim citation verification
    explain.py            Template explanations + next steps
  data/
    agreements.json       4 representative agreements (38 clauses)
    ai_policies.json      3 representative AI policies (17 clauses)
    benchmark.json/.csv   40 scenarios with labels, evidence, uncertainty
    README.md             Data provenance + pattern sources
  tests/
    validate_benchmark.py Benchmark validation gate
  results/                Saved evaluation output (regenerate with evaluate.py)
  report/FINAL_REPORT.md  Final report (submittable draft)
  presentation/DEMO_SCRIPT.md  5-minute demo script + Q&A
```

## Limitations (summary; full discussion in the report)

- Representative paraphrased clauses, not real agreements; no legal
  weight. Labels reflect the authors' reading of those clauses, with no
  independent annotator or inter-annotator agreement.
- Same 40 scenarios were used during development and evaluation (no
  held-out split), so accuracy is optimistic.
- The shipped numbers come from the fallback scorer, not the LLM.
- Small-n safety claims: 0/10 false approvals does not prove general
  safety. Confidence scores are uncalibrated heuristics.
- TF-IDF retrieval with a crude stemmer misses paraphrases that dense
  embeddings would catch; negation ("no data values") can fool cue
  patterns (see failure S08 in Setting A).

## Reproducibility

- `python tests/validate_benchmark.py` must pass before evaluation.
- `python evaluate.py` is deterministic (no sampling, no network calls)
  with the default backends and rewrites all files in `results/`.
- All thresholds, cue patterns, and stance mappings are constants in
  `dua_guard/classify.py` with explanatory comments; nothing is tuned
  per-scenario and no benchmark label is read by the pipeline.
