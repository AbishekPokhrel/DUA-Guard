"""DUA Guard -- Streamlit application.

Run:  streamlit run app.py

The app has four views, each carrying the disclaimer that DUA Guard
gives an evidence-based interpretation of representative clause sets,
not legal or authoritative institutional compliance advice:

  1. Check an action -- the main workflow: pick a dataset and AI tool,
     describe a proposed action, get a decision with verbatim citations.
  2. Benchmark browser -- inspect the 40 AI-DUA-Bench scenarios, their
     ground-truth labels, evidence clauses, and documented uncertainty.
  3. Evaluation results -- the saved results of `python evaluate.py`
     (run the evaluation first; the app never invents numbers).
  4. Agreements & policies -- read the representative clause sets.

No API key is required. By default the app uses the transparent
evidence-grounded fallback engine and TF-IDF retrieval, both of which
run fully offline. The free local FLAN-T5 model and embedding retrieval
are optional and clearly labelled when active (see the sidebar).
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from dua_guard import DISCLAIMER, LABELS
from dua_guard.classify import predict
from dua_guard.data_loader import (
    build_clause_index,
    extract_clauses_from_text,
    get_agreement,
    get_clauses,
    get_policy,
    load_agreements,
    load_benchmark,
    load_policies,
)
from dua_guard.explain import (
    build_explanation,
    format_citations_for_display,
    recommended_next_steps,
)
from dua_guard.llm_client import MODEL_NAME, get_engine_info

PROJECT_ROOT = Path(__file__).resolve().parent

LABEL_COLORS = {
    "Allowed": "#1a7f37",
    "Requires Approval": "#9a6700",
    "Prohibited": "#cf222e",
    "Insufficient Information / Requires Human Review": "#57606a",
}

SETTING_HELP = {
    "A": "Question only: no clauses retrieved, no citations. Baseline.",
    "B": "Retrieved clauses + classification, top clause cited without verification.",
    "C": "Full DUA Guard: retrieval + citation verification + conservative abstention. Recommended.",
}

EXAMPLE_ACTIONS = {
    "Allowed example -- public-use upload to a cloud model": (
        "A researcher uploads the public-use dataset file to a cloud large "
        "language model to explore variable distributions and draft summary tables.",
        "open-public-use",
        "cloud-llm",
    ),
    "Prohibited example -- patient records pasted into a cloud model": (
        "A researcher pastes de-identified patient-level intensive care records "
        "into a cloud large language model and asks for a summary of each patient's course.",
        "physionet-credentialed",
        "cloud-llm",
    ),
    "Insufficient example -- underspecified cloud upload": (
        "A researcher plans to upload a file derived from credentialed health data "
        "to a cloud large language model; the description does not state whether the "
        "file contains individual-level records, aggregate statistics only, or code "
        "with no data values.",
        "physionet-credentialed",
        "cloud-llm",
    ),
}


@st.cache_data
def cached_agreements():
    return load_agreements()


@st.cache_data
def cached_policies():
    return load_policies()


@st.cache_data
def cached_benchmark():
    return load_benchmark()


@st.cache_data
def cached_clause_index():
    return build_clause_index()


def show_disclaimer() -> None:
    st.warning(DISCLAIMER, icon="⚠️")


def decision_badge(label: str) -> None:
    color = LABEL_COLORS.get(label, "#57606a")
    st.markdown(
        f"<div style='background:{color};color:white;padding:10px 16px;"
        f"border-radius:8px;font-size:20px;font-weight:600;'>{label}</div>",
        unsafe_allow_html=True,
    )


def read_uploaded_file(uploaded) -> str:
    """Read an uploaded .txt file, or a PDF when pypdf is installed."""
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except Exception:
            st.error(
                "PDF upload needs the optional `pypdf` package, which is not "
                "installed. Paste the agreement text instead, or install pypdf."
            )
            return ""
        reader = PdfReader(uploaded)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded.read().decode("utf-8", errors="replace")


def check_action_view(dataset_id: str, policy_id: str, top_k: int, setting: str) -> None:
    show_disclaimer()
    st.subheader("Describe the proposed action")
    st.caption(
        "Be specific: what data, what tool, where data will go, who receives "
        "outputs, and whether approval was obtained. Vague descriptions get "
        "Insufficient Information -- that is the system working as designed."
    )

    example = st.selectbox(
        "Or load an example:",
        ["(write my own action)"] + list(EXAMPLE_ACTIONS),
        index=0,
    )
    default_action = ""
    if example != "(write my own action)":
        default_action = EXAMPLE_ACTIONS[example][0]
        st.caption(
            "Example loaded. The sidebar dataset/tool selections still apply; "
            f"this example was written for {EXAMPLE_ACTIONS[example][1]} + "
            f"{EXAMPLE_ACTIONS[example][2]}."
        )
    action = st.text_area(
        "Proposed action",
        value=default_action,
        height=120,
        placeholder="Example: Upload the row-level restricted survey file to a cloud chatbot to clean the data.",
    )

    with st.expander("Optional: check against your own agreement text"):
        st.caption(
            "Paste agreement text or upload a .txt file (PDF needs optional "
            "pypdf). Your text is split into clauses and added to the selected "
            "agreement's clauses for this check only. Custom clauses are "
            "labelled CUSTOM and their effect is treated as unknown, so the "
            "system abstains more often -- it will not pretend to understand "
            "text it has not classified. Your text is not stored."
        )
        custom_text = st.text_area("Paste agreement text", height=120)
        uploaded = st.file_uploader("Or upload a .txt (or .pdf) agreement", type=["txt", "pdf"])
        uploaded_text = read_uploaded_file(uploaded)

    use_llm = st.checkbox(
        f"Try the free local LLM ({MODEL_NAME}) instead of the fallback engine",
        value=False,
        help="Only enable this on a machine with transformers + torch installed and "
        "enough memory. The first run downloads the model (~1 GB). If the model "
        "cannot load, the fallback engine runs instead and the result says so.",
    )

    if st.button("Check compliance", type="primary"):
        if not action.strip():
            st.error("Please describe a proposed action first.")
            return
        clauses = get_clauses(dataset_id, policy_id)
        custom_clauses = extract_clauses_from_text(custom_text or uploaded_text)
        if custom_clauses:
            clauses = clauses + custom_clauses
            st.info(
                f"Added {len(custom_clauses)} custom clause(s) from your text "
                "(labelled CUSTOM, effect unknown)."
            )
        with st.spinner("Retrieving clauses and checking compliance..."):
            result = predict(
                action,
                clauses,
                setting=setting,
                top_k=top_k,
                use_llm=use_llm,
            )
        st.divider()
        decision_badge(result["label"])
        st.caption(
            f"Confidence (heuristic, not a probability): {result['confidence']:.2f}  |  "
            f"Setting {setting}  |  Engine: {result['engine']}  |  "
            f"Retrieval: {result.get('retrieval_backend') or 'none (Setting A)'}"
        )
        if result.get("llm_fallback"):
            st.warning(
                "The local LLM was requested but unavailable or unparseable for "
                "this case, so the fallback engine produced this result."
            )
        if result.get("override_reason"):
            st.info(f"Conservative abstention applied: {result['override_reason']}.")
        st.write(build_explanation(result))

        st.subheader("Evidence (verbatim quotes)")
        citations = format_citations_for_display(result)
        if citations:
            for citation in citations:
                st.markdown(f"**{citation['clause_id']} -- {citation['title']}**")
                st.markdown(f"> {citation['quote']}")
        else:
            st.write("No citation. Setting A produces no citations by design; "
                     "in Setting C, no citation means no clause could verify a decision.")

        st.subheader("Recommended next steps")
        agreement_name = get_agreement(dataset_id)["name"]
        for step in recommended_next_steps(result, agreement_name):
            st.write(f"- {step}")

        with st.expander("Retrieved clauses and scores (for inspection)"):
            st.write(
                "Retrieval scores are TF-IDF cosine similarities (or embedding "
                "similarities when that backend is enabled). Label scores combine "
                "retrieval similarity over clause stances with generic cue patterns; "
                "see HOW_IT_WORKS.md."
            )
            for item in result["retrieved"]:
                st.write(
                    f"- **{item['clause_id']}** [{item['stance']}] "
                    f"similarity {item['similarity']:.3f} -- {item['title']}"
                )
            st.bar_chart(result["scores"])
            st.write("Matched cue patterns:", ", ".join(result["matched_cues"]) or "none")
    show_disclaimer()


def benchmark_view() -> None:
    show_disclaimer()
    st.subheader("AI-DUA-Bench: 40 scenarios")
    st.caption(
        "Synthetic scenarios only -- no real participant data. Ground-truth labels "
        "were assigned by the authors from the representative clause sets and "
        "were never copied from a model. Uncertainty documents what is missing "
        "or ambiguous in each case."
    )
    benchmark = cached_benchmark()
    scenarios = benchmark["scenarios"]
    index = cached_clause_index()
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox(
            "Filter by category",
            ["All"] + sorted({s["category"] for s in scenarios}),
        )
    with col2:
        label = st.selectbox("Filter by ground-truth label", ["All"] + LABELS)
    shown = [
        s for s in scenarios
        if (category == "All" or s["category"] == category)
        and (label == "All" or s["ground_truth"] == label)
    ]
    st.write(f"Showing {len(shown)} of {len(scenarios)} scenarios.")
    for scenario in shown:
        with st.expander(f"{scenario['id']} -- {scenario['category']} -- {scenario['ground_truth']}"):
            st.write(scenario["action"])
            st.write(f"**Dataset:** {scenario['dataset']}  |  **AI tool:** {scenario['ai_tool']}")
            st.write(f"**Ground truth:** {scenario['ground_truth']}")
            st.write(f"**Rationale:** {scenario['rationale']}")
            st.write(
                f"**Uncertainty ({scenario['uncertainty']}):** {scenario['uncertainty_note']}"
            )
            st.write("**Evidence clauses:**")
            for clause_id in scenario["evidence_clause_ids"]:
                clause = index[clause_id]
                st.markdown(f"- **{clause_id}** [{clause['stance']}] {clause['text']}")


def evaluation_view() -> None:
    show_disclaimer()
    st.subheader("Evaluation results (saved run)")
    results_path = PROJECT_ROOT / "results" / "evaluation_results.json"
    if not results_path.exists():
        st.error(
            "No saved evaluation found. Run `python evaluate.py` first, then "
            "reload this tab. This app never invents evaluation numbers."
        )
        return
    with open(results_path, encoding="utf-8") as handle:
        results = json.load(handle)
    st.caption(
        f"Benchmark: {results['benchmark']} -- {results['n_scenarios']} scenarios, "
        f"top-k={results['top_k']}. Engine: {results['engine_info']['engine']}."
    )
    st.write(results["engine_info"]["reason"])
    import pandas as pd

    rows = []
    for setting in ("A", "B", "C"):
        m = results["settings"][setting]
        rows.append(
            {
                "Setting": setting,
                "Accuracy": m["accuracy"],
                "False approval rate": m["false_approval_rate"],
                "Unsafe rate": m["unsafe_rate"],
                "Citation accuracy": m["citation_accuracy"],
                "Abstention rate": m["abstention_rate"],
                "Committed accuracy": m["committed_accuracy"],
                "Macro-F1": m["macro_f1"],
                "Mean latency (ms)": m["mean_latency_ms"],
            }
        )
    table = pd.DataFrame(rows).set_index("Setting")
    st.dataframe(table)
    st.bar_chart(table[["Accuracy", "False approval rate", "Citation accuracy"]].fillna(0))
    st.caption(
        "False approval rate is the primary safety metric: true Prohibited "
        "cases predicted as Allowed / all true Prohibited cases. Citation "
        "accuracy is null for Setting A because it produces no citations."
    )

    setting_choice = st.selectbox("Inspect failures for setting:", ["A", "B", "C"], index=2)
    failures = results["settings"][setting_choice]["failures"]
    st.write(f"Setting {setting_choice} errors: {len(failures)} of {results['n_scenarios']}.")
    for failure in failures:
        with st.expander(
            f"{failure['scenario_id']}: true={failure['true_label']}, "
            f"predicted={failure['predicted']}"
        ):
            st.write(failure["action"])
            st.write(f"Likely cause: {failure['likely_cause']}")
            st.write(f"Cited: {failure['cited_clause_ids']}  |  "
                     f"Evidence: {failure['evidence_clause_ids']}")
            if failure.get("override_reason"):
                st.write(f"Override: {failure['override_reason']}")

    st.subheader("Per-category accuracy")
    cat_rows = []
    for setting in ("A", "B", "C"):
        for category, acc in results["settings"][setting]["per_category_accuracy"].items():
            cat_rows.append({"Setting": setting, "Category": category, "Accuracy": acc})
    st.dataframe(pd.DataFrame(cat_rows))


def agreements_view() -> None:
    show_disclaimer()
    st.subheader("Representative agreements and AI policies")
    st.caption(
        "These clause sets are paraphrased from common public DUA patterns for "
        "classroom evaluation. They are NOT verbatim legal documents and NOT "
        "the actual agreements of ICPSR, PhysioNet, or UK Biobank. Pattern "
        "sources are documented in data/agreements.json."
    )
    for agreement in cached_agreements():
        with st.expander(f"{agreement['name']} ({agreement['id']})"):
            st.write(agreement["description"])
            for clause in agreement["clauses"]:
                st.markdown(
                    f"- **{clause['id']}** [{clause['stance']}] **{clause['title']}**: "
                    f"{clause['text']}"
                )
    for policy in cached_policies():
        with st.expander(f"{policy['name']} ({policy['id']})"):
            st.write(policy["description"])
            for clause in policy["clauses"]:
                st.markdown(
                    f"- **{clause['id']}** [{clause['stance']}] **{clause['title']}**: "
                    f"{clause['text']}"
                )


def main() -> None:
    st.set_page_config(page_title="DUA Guard", layout="wide")
    st.title("DUA Guard: compliance-aware AI for Data Use Agreements")
    st.caption(
        "Two-person group project by Abishek Pokhrel and Himal Aryal. "
        "Free/local models only -- no API key, no paid services."
    )

    agreements = cached_agreements()
    policies = cached_policies()
    st.sidebar.header("Configuration")
    dataset_id = st.sidebar.selectbox(
        "Dataset agreement",
        [a["id"] for a in agreements],
        format_func=lambda i: next(a["name"] for a in agreements if a["id"] == i),
    )
    policy_id = st.sidebar.selectbox(
        "AI tool / policy",
        [p["id"] for p in policies],
        format_func=lambda i: next(p["name"] for p in policies if p["id"] == i),
    )
    top_k = st.sidebar.slider("Retrieved clauses (top-k)", 3, 8, 5)
    setting = st.sidebar.radio(
        "Pipeline setting",
        ["A", "B", "C"],
        index=2,
        format_func=lambda s: f"Setting {s} -- {SETTING_HELP[s]}",
    )

    st.sidebar.header("Engine status")
    engine_info = get_engine_info(check_llm=False)
    st.sidebar.write(f"**Classification engine:** {engine_info['engine']}")
    st.sidebar.caption(engine_info["reason"])
    st.sidebar.write("**Retrieval backend:** TF-IDF (NumPy, offline by default). "
                     "Set DUA_RETRIEVAL_BACKEND=embeddings or =sklearn to try the "
                     "optional backends; the active backend is shown with each result.")
    st.sidebar.caption(
        "Representative clause sets only. Not legal advice. See the disclaimer "
        "on every tab."
    )

    tabs = st.tabs(
        ["Check an action", "Benchmark browser", "Evaluation results", "Agreements & policies"]
    )
    with tabs[0]:
        check_action_view(dataset_id, policy_id, top_k, setting)
    with tabs[1]:
        benchmark_view()
    with tabs[2]:
        evaluation_view()
    with tabs[3]:
        agreements_view()


if __name__ == "__main__":
    main()
