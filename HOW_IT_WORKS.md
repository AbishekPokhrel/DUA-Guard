# How DUA Guard works (plain English)

This file explains the code so you can present and defend it. If you
read nothing else, read the pipeline walkthrough below and the
"Questions you may be asked" section at the end.

## The idea in one paragraph

A researcher describes something they want to do with AI ("upload this
file to a chatbot", "let an agent clean the data"). DUA Guard looks up
the relevant sentences (clauses) from the dataset's agreement and the AI
tool's policy, decides Allowed / Requires Approval / Prohibited /
Insufficient Information based on those clauses, shows the exact clause
wording it relied on, and refuses to guess when the information is not
there. It never answers from memory alone: no clause, no decision.

## The pipeline, step by step

### 1. Pick the governing documents (`data_loader.py`)

The app loads two JSON files: `data/agreements.json` (4 agreements, 38
clauses) and `data/ai_policies.json` (3 AI policies, 17 clauses). Each
clause has an ID (e.g. `PHYS-03`), a title, the clause text, a stance,
and topics. The stance says what the clause *does*:

- `prohibition` — "must not do X"
- `permission` — "you may do X"
- `approval_required` — "you need approval before X"
- `condition` — "X only under conditions" (depends on context)
- `obligation` — "you must do Y" (e.g. report, delete, verify)

When the user picks a dataset and an AI tool, retrieval is scoped to
exactly those two documents (13–16 clauses). The system can never cite a
clause from a different agreement. This scoping is what lets it say
Allowed for uploading *public-use* data but Prohibited for uploading
*restricted* data: the same action retrieves different governing clauses.

If the user pastes their own agreement text, `extract_clauses_from_text`
splits it into paragraphs/numbered items labelled `CUSTOM-01`, ... with
stance `condition` (unknown effect). The system is honest that it has
not classified custom text, so it abstains more often.

### 2. Retrieve the relevant clauses (`retrieval.py`)

This is the RAG retrieval step. The user's action text is turned into a
bag of words (lowercased, stopwords removed, lightly stemmed so
"uploads"/"uploading" match "upload"), and each clause is scored by
TF-IDF cosine similarity: words that are rare across the clause set
count more than common words. The top-k (default 5) clauses are kept.

Two optional backends exist: scikit-learn TF-IDF and dense embeddings
(`all-MiniLM-L6-v2`). They are only used when explicitly requested with
`DUA_RETRIEVAL_BACKEND`, because the embedding model must be downloaded.
The backend that actually ran is recorded with every result — the
shipped evaluation used the NumPy TF-IDF backend.

Limitation to know: TF-IDF matches words, not meanings. A paraphrase
with different words can be missed, and a clause sharing vocabulary but
not meaning can rank highly. That is one reason later stages verify and
abstain instead of trusting retrieval blindly.

### 3. Score the labels (`classify.py`)

Two sources of evidence are combined into a score for each of the four
labels:

- **Retrieval scores.** Each retrieved clause contributes its similarity
  × its stance signal: a prohibition clause adds to Prohibited, a
  permission clause adds to Allowed, an approval clause adds to Requires
  Approval, and condition/obligation clauses add weakly to Requires
  Approval and Insufficient Information (because they make the answer
  depend on context).
- **Cue scores.** Generic, human-readable patterns in the action text,
  e.g. "upload/paste/send + individual-level data" (Prohibited),
  "without seeking approval" (Requires Approval), "no data values" /
  "only variable names" (Allowed), "does not state" / "unclear"
  (Insufficient Information). These are *generic* DUA reasoning patterns
  written from the clause language — they never mention scenario IDs and
  never read benchmark labels.

When the free local LLM (FLAN-T5) is available and enabled, it replaces
the scorer: it receives the retrieved clauses and the action, and must
answer in strict JSON. If its output cannot be parsed, that case falls
back to the scorer and the fallback is counted. In the shipped
evaluation the LLM was unavailable (no transformers/torch), so all
numbers come from the scorer — the report and README say this plainly.

### 4. Verify the citation (`verify.py`)

The system picks the highest-ranked retrieved clause whose stance
supports the predicted label, and quotes its text *verbatim*. A quote
only counts if, after normalizing case/whitespace/punctuation, it is
found word-for-word inside a retrieved clause. Short fragments never
count. If the LLM path produced a quote, the same check applies: an
invented or paraphrased quote is dropped, not fixed up.

Special rule for Prohibited: a restrictive condition ("access only on
approved systems") also supports Prohibited, because doing it outside
the stated conditions is prohibited by that clause. Allowed is stricter:
it needs a permission clause with similarity at least 0.12, and no
strong prohibition evidence in the retrieved set.

### 5. Decide whether to abstain (`classify.py`, Setting C only)

Setting C applies conservative rules after scoring:

- Top score too low (< 0.45) or margin too close (< 0.10) → Insufficient.
- An uncertainty phrase ("does not state", "unclear") with a close
  margin (< 0.35) → Insufficient.
- No clause supports the candidate label → Insufficient.
- Allowed without a verified permission citation, or with conflicting
  prohibition evidence → Insufficient.

Setting B skips all of this (always commits, cites the top clause
unverified). Setting A skips retrieval entirely (cue scores only, no
citations). Comparing the three settings is the evaluation: it shows
what retrieval adds (evidence + citations) and what verification +
abstention add (safety behaviour), separately.

### 6. Explain and recommend (`explain.py`)

The explanation is built from a template filled with the verified
citation IDs — the system cannot invent clause language in the
explanation, because the only clause text it prints is the verified
quote. Next steps are per-label checklists (e.g. for Requires Approval:
identify the approver, submit an amendment, wait for written approval).
The disclaimer appears on every view of the app and in the report.

## A worked example

Action: *"A researcher pastes de-identified patient-level intensive care
records into a cloud large language model."* Dataset: PhysioNet-style.
Tool: cloud LLM.

1. Retrieval returns PHYS-03 ("must not be shared with any third party,
   including sending the data through APIs") at high similarity, plus
   PHYS-04, CLOUD-03, and others.
2. Scores: Prohibited leads — prohibition clauses score highly, and the
   disclosure cue ("pastes" + "patient-level") fires.
3. Citation: PHYS-03 supports Prohibited and is quoted verbatim.
4. No abstention rule fires: the top score and margin are comfortable,
   and no uncertainty phrase is present.
5. Output: **Prohibited**, with the PHYS-03 quote, an explanation, and
   next steps (do not do this; consider aggregate-only alternatives).

Contrast with: *"...plans to upload a file derived from credentialed
health data; the description does not state whether the file contains
individual-level records, aggregate statistics only, or code."* The
uncertainty phrase fires, the margin is close, and Setting C abstains
with Insufficient Information — the correct behaviour, because the
answer genuinely depends on what the file contains.

## File-by-file map

| File | What it does | Key functions/classes |
|---|---|---|
| `app.py` | Streamlit UI: 4 tabs, sidebar config, result display | `main`, `check_action_view`, `benchmark_view`, `evaluation_view`, `agreements_view` |
| `evaluate.py` | Runs A/B/C on all 40 scenarios, computes metrics, writes results | `evaluate_setting`, `main` |
| `dua_guard/data_loader.py` | Loads JSON data; builds clause index; splits custom DUA text | `get_clauses`, `build_clause_index`, `extract_clauses_from_text`, `validate_benchmark` |
| `dua_guard/retrieval.py` | TF-IDF + optional embedding retrieval | `NumpyTfidfRetriever`, `build_retriever`, `tokenize` |
| `dua_guard/llm_client.py` | Local FLAN-T5 client; prompt building; JSON parsing; availability reporting | `LocalLLM`, `build_prompt`, `parse_llm_output`, `get_engine_info`, `classify_with_llm` |
| `dua_guard/classify.py` | Scoring, 3 settings, abstention rules | `cue_scores`, `retrieval_scores`, `predict` |
| `dua_guard/verify.py` | Verbatim citation checking | `normalize_text`, `quote_matches_clause`, `verify_citation` |
| `dua_guard/explain.py` | Template explanations + next steps | `build_explanation`, `recommended_next_steps` |
| `tests/validate_benchmark.py` | Data validation gate | `main` |

## Questions you may be asked (with honest answers)

**"Is this really using an LLM?"** — The code supports a free local LLM
(FLAN-T5) with prompting and structured JSON output, and the app can try
it. But the shipped evaluation numbers were produced by the transparent
fallback scorer because transformers/torch were not installed in the
build environment. We record which engine ran and never present fallback
numbers as LLM numbers. The GenAI components that did run are retrieval,
structured extraction, multi-stage reasoning, verification, and
template-grounded explanation — plus the LLM path as implemented,
runnable code.

**"Why is accuracy the same for all three settings?"** — Because the
scenarios were written with clear decision patterns, cue patterns alone
get most labels right. The settings differ where it matters for safety:
citations (A has none, B/C cite), abstention on underspecified cases
(B commits, C abstains), and committed accuracy (C highest). Accuracy
alone would hide those differences, which is why the report leads with
false approval rate and citation accuracy.

**"Are the labels ground truth?"** — They are the authors' reading of
representative paraphrased clauses, not authoritative legal
interpretations. There was no second annotator. The benchmark measures
whether the pipeline agrees with that documented reading, including its
uncertainty — it does not prove what a data provider or court would say.

**"Why abstain instead of guessing?"** — In compliance, a wrong Allowed
is worse than no answer. Abstention routes the case to a human, which is
the safe default. The cost is visible too: Setting C over-abstains once
(S08), and the report discusses that tradeoff openly.

**"What would you do with more time?"** — Dense embedding retrieval,
independent second annotator with agreement measurement, a held-out test
split, real (permission-cleared) agreement excerpts alongside the
representative sets, and an LLM evaluation run on a machine with
sufficient memory — in that order.
