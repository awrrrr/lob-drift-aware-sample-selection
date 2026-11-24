"""Streamlit viewer for the experiment logs. Reads runs.db (plus two finished
result files: results/revision/bootstrap.csv and results/learning_curve.png).

This is a viewer, not the contribution: every number shown comes straight from
runs.db (written by the harness and src/evaluate.py) or those result files. It
never loads the raw data or retrains. The only arithmetic is differences between
stored values and per-class F1 from stored precision/recall. Run with:

    .venv/bin/python -m streamlit run app/dashboard.py
"""

import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "runs.db"
BOOTSTRAP_CSV = ROOT / "results" / "revision" / "bootstrap.csv"
LEARNING_CURVE_PNG = ROOT / "results" / "learning_curve.png"
CHANCE_F1 = 1 / 3   # random guessing over 3 classes (test labels are ~35/32/33%)
CLASSES = ["Up", "Stat", "Down"]
MARK = "#4C78A8"     # one hue for the mean; seeds in neutral gray
CHANCE_COLOR = "#F58518"  # reference line, distinct from the data hue

POLICY_ORDER = ["A", "B", "C", "D", "E", "E_minus_audit",
                "E_drift", "E_minus_cluster", "E_drift_minus_cluster"]
POLICY_LABEL = {"A": "A fixed", "B": "B random", "C": "C uncertainty",
                "D": "D density×unc", "E": "E proposed", "E_minus_audit": "E−audit",
                "E_drift": "E_drift (revision)", "E_minus_cluster": "E−cluster (revision)",
                "E_drift_minus_cluster": "E_drift−cluster (revision)"}
# Runs shown: the original step-5 runs ('main') and scored revision runs.
# Grid and check runs ('revision_grid:*', 'revision_check') are tuning/diagnostic
# runs that are never test-scored, so they are left out.
SHOWN_TAGS = ("main", "revision")


# --------------------------------------------------------------------------- #
# Pure data helpers (no Streamlit; unit-tested in tests/test_dashboard.py)
# --------------------------------------------------------------------------- #

def load_tables(db_path):
    """Read the four tables into DataFrames. Missing tables come back empty."""
    con = sqlite3.connect(str(db_path))
    out = {}
    for name in ["runs", "selections", "updates", "metrics"]:
        try:
            out[name] = pd.read_sql(f"SELECT * FROM {name}", con)
        except Exception:
            out[name] = pd.DataFrame()
    con.close()
    runs = out["runs"]
    if not runs.empty and "tag" in runs.columns:
        keep = runs.tag.fillna("main").isin(SHOWN_TAGS)
        out["runs"] = runs[keep].reset_index(drop=True)
        ids = set(out["runs"].run_id)
        for name in ["selections", "updates", "metrics"]:
            if not out[name].empty:
                out[name] = out[name][out[name].run_id.isin(ids)].reset_index(drop=True)
    return out


def test_metric(metrics, runs, condition, metric):
    """Per-(policy, seed) values for a test metric, as a merged DataFrame."""
    m = metrics[(metrics.split == "test") & (metrics.condition == condition)
                & (metrics.metric == metric)]
    return m.merge(runs[["run_id", "policy", "seed"]], on="run_id")


def summarize(tables):
    """One row per policy: mean +/- std over seeds of the headline metrics."""
    runs, metrics = tables["runs"], tables["metrics"]
    if runs.empty or metrics.empty:
        return pd.DataFrame()
    rows = {}
    for cond, mt, key in [("high_vol", "macro_f1", "high_vol_F1"),
                          ("all", "macro_f1", "overall_F1"),
                          ("low_vol", "macro_f1", "low_vol_F1"),
                          ("all", "accuracy", "accuracy")]:
        d = test_metric(metrics, runs, cond, mt)
        if d.empty:
            continue
        g = d.groupby("policy").value
        rows[key + "_mean"] = g.mean()
        rows[key + "_std"] = g.std().fillna(0.0)
    df = pd.DataFrame(rows)
    order = [p for p in POLICY_ORDER if p in df.index]
    return df.reindex(order)


def ordered_policies(runs):
    present = set(runs.policy) if not runs.empty else set()
    return [p for p in POLICY_ORDER if p in present]


def fixed_model_vs_chance(tables):
    """Policy A (V1, never updated) against chance: one row per class plus macro.

    Per-class F1 = 2PR/(P+R) from the stored test precision/recall. Returns an
    empty frame if policy A has not been scored.
    """
    runs, metrics = tables["runs"], tables["metrics"]
    if runs.empty or metrics.empty:
        return pd.DataFrame()
    ids = runs[runs.policy == "A"].run_id
    m = metrics[metrics.run_id.isin(ids) & (metrics.split == "test")
                & (metrics.condition == "all")]
    if m.empty:
        return pd.DataFrame()
    v = m.groupby("metric").value.mean()
    rows = []
    for c in CLASSES:
        p, r = v.get(f"precision_{c}"), v.get(f"recall_{c}")
        if p is None or r is None:
            continue
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        rows.append({"class": c, "precision": p, "recall": r, "F1": f1})
    if "macro_f1" in v:
        rows.append({"class": "macro", "F1": v["macro_f1"]})
    out = pd.DataFrame(rows)
    out["F1_minus_chance_pp"] = (out.F1 - CHANCE_F1) * 100
    return out


def gain_over_fixed(tables, condition="high_vol"):
    """Per-run test macro-F1 minus policy A's, in percentage points.

    One row per (policy, seed) for every updating policy. A is the reference
    (gain 0 by construction) and is left out.
    """
    d = test_metric(tables["metrics"], tables["runs"], condition, "macro_f1")
    if d.empty or "A" not in set(d.policy):
        return pd.DataFrame()
    ref = d[d.policy == "A"].value.mean()
    out = d[d.policy != "A"][["policy", "seed", "value"]].copy()
    out["gain_pp"] = (out.value - ref) * 100
    out["label"] = out.policy.map(lambda p: POLICY_LABEL.get(p, p))
    return out.reset_index(drop=True)


def load_bootstrap(path, condition="high_vol"):
    """E_drift-vs-others bootstrap CIs (results/revision/bootstrap.csv), if present."""
    if not Path(path).exists():
        return pd.DataFrame()
    b = pd.read_csv(path)
    return b[b.condition == condition].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #

def show_model_vs_chance(tables):
    """Section 1: the base model clearly learns. Selection is a separate question."""
    st.subheader("1 · Did the model learn? Yes: V1 against chance")
    d = fixed_model_vs_chance(tables)
    if d.empty:
        st.info("Policy A (fixed V1) has no test metrics yet.")
        return
    macro = d[d["class"] == "macro"].F1.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("V1 test macro-F1 (all test windows)", f"{macro:.4f}")
    c2.metric("Random guessing, 3 classes", f"{CHANCE_F1:.3f}")
    c3.metric("V1 above chance", f"+{(macro - CHANCE_F1) * 100:.1f} pp")

    per_class = d[d["class"] != "macro"]
    bars = alt.Chart(per_class).mark_bar(color=MARK, cornerRadiusEnd=4, size=16).encode(
        y=alt.Y("class:N", sort=CLASSES, title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 0.6]), title="test F1 per class, fixed V1 (orange dashed = chance)"),
        tooltip=["class", alt.Tooltip("F1:Q", format=".3f"),
                 alt.Tooltip("precision:Q", format=".3f"), alt.Tooltip("recall:Q", format=".3f")])
    chance = alt.Chart(pd.DataFrame({"F1": [CHANCE_F1]})).mark_rule(
        color=CHANCE_COLOR, strokeWidth=2, strokeDash=[5, 3]).encode(x="F1:Q")
    st.altair_chart((bars + chance).properties(height=150), use_container_width=True)
    st.caption("Orange dashed line = chance (1/3). Every class is above it; a model that always "
               "predicts one class scores only ≈ 0.17 macro-F1. Bars start at zero. "
               "Per-class F1 = 2PR/(P+R) from the stored test precision and recall.")


def show_gain_over_fixed(tables):
    """Section 2: what updating adds over the fixed model, per policy and seed."""
    st.subheader("2 · What does updating add? Gain over the fixed model (A)")
    cond = st.radio("Test windows", ["high_vol", "all", "low_vol"], horizontal=True,
                    format_func={"high_vol": "high-volatility (primary)", "all": "all",
                                 "low_vol": "low-volatility"}.get)
    g = gain_over_fixed(tables, cond)
    if g.empty:
        st.info("Needs test metrics for policy A and at least one updating policy.")
        return
    order = [POLICY_LABEL.get(p, p) for p in POLICY_ORDER if p in set(g.policy)]
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    seeds = alt.Chart(g).mark_circle(size=60, color="#999", opacity=0.7).encode(
        y=y, x=alt.X("gain_pp:Q", title="test macro-F1 minus fixed model A (percentage points)"),
        tooltip=["label", "seed", alt.Tooltip("gain_pp:Q", format="+.3f", title="gain (pp)"),
                 alt.Tooltip("value:Q", format=".4f", title="macro-F1")])
    means = alt.Chart(g).mark_tick(color=MARK, thickness=3, size=22).encode(
        y=y, x="mean(gain_pp):Q",
        tooltip=["label", alt.Tooltip("mean(gain_pp):Q", format="+.3f", title="mean gain (pp)")])
    zero = alt.Chart(pd.DataFrame({"gain_pp": [0.0]})).mark_rule(
        color="#888", strokeDash=[4, 3]).encode(x="gain_pp:Q")
    st.altair_chart((zero + seeds + means).properties(height=36 * len(order)),
                    use_container_width=True)
    lo, hi = g.gain_pp.min(), g.gain_pp.max()
    read = ("On high-volatility windows every seed of every policy is right of zero: "
            "updating helps there (about +0.2 to +0.3 pp), and the policies overlap, so "
            "which samples are picked does not matter at B = 50."
            if cond == "high_vol" else
            "Here the gains hug zero for every policy: updating barely changes the model "
            "on these windows, whichever samples are picked.")
    st.caption(f"Gray dots = individual seeds; blue tick = mean over seeds; dashed line = no "
               f"change from the fixed model. Gains range from {lo:+.2f} to {hi:+.2f} "
               f"percentage points, so the axis is in pp to make them readable. {read}")


def show_bootstrap():
    """Section 3: E_drift vs each other policy with 95% bootstrap intervals."""
    b = load_bootstrap(BOOTSTRAP_CSV)
    if b.empty:
        return
    st.subheader("3 · Is any difference significant? E_drift vs others, 95% CI")
    b = b.assign(other=b.comparison.str.replace("E_drift - ", "", regex=False))
    b["label"] = b.other.map(lambda p: POLICY_LABEL.get(p, p))
    b["significant"] = (b.ci95_low_pp > 0) | (b.ci95_high_pp < 0)
    order = list(b.label)
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    tip = ["label", alt.Tooltip("diff_pp:Q", format="+.3f", title="difference (pp)"),
           alt.Tooltip("ci95_low_pp:Q", format="+.3f", title="CI low"),
           alt.Tooltip("ci95_high_pp:Q", format="+.3f", title="CI high"), "significant"]
    bars = alt.Chart(b).mark_rule(strokeWidth=2, color=MARK).encode(
        y=y, x=alt.X("ci95_low_pp:Q", title="E_drift minus each policy: high-vol macro-F1 difference (pp), 95% CI"),
        x2="ci95_high_pp:Q", tooltip=tip)
    dots = alt.Chart(b).mark_circle(size=70, color=MARK, opacity=1).encode(
        y=y, x="diff_pp:Q", tooltip=tip)
    zero = alt.Chart(pd.DataFrame({"diff_pp": [0.0]})).mark_rule(
        color="#888", strokeDash=[4, 3]).encode(x="diff_pp:Q")
    st.altair_chart((zero + bars + dots).properties(height=36 * len(b)),
                    use_container_width=True)
    n = int(b.n_windows.iloc[0])
    st.caption(f"Paired bootstrap over the {n} high-volatility test windows (2,000 "
               "resamples; results/revision/bootstrap.csv). An interval crossing the dashed "
               "zero line is not significant. Only the comparison with the fixed model A is "
               "clear of zero: updating helps, and which samples are picked does not.")


def show_learning_curve():
    """Section 4: why selection cannot matter much at this budget."""
    if not LEARNING_CURVE_PNG.exists():
        return
    st.subheader("4 · Why so flat? The budget is small next to the training set")
    st.image(str(LEARNING_CURVE_PNG), use_container_width=True)
    st.caption("Random stream samples added to the 178k-sample training set (from "
               "scripts/failure_analysis.py; the y-axis is zoomed). The model does keep "
               "improving with more data: 0.4335 → 0.4432 overall with all 93k stream "
               "samples. The budget used, 93 windows × 50 = 4,650 samples (dashed line), "
               "adds only 2.6% to the base, so any 50 samples per window give about the "
               "same model.")


def main():
    st.set_page_config(page_title="LOB drift-aware selection — runs viewer", layout="wide")
    st.title("LOB drift-aware sample selection — experiment viewer")
    st.caption("A viewer over runs.db and the finished results files. Nothing here "
               "retrains; the only arithmetic is differences between stored values.")

    db_path = st.sidebar.text_input("runs.db path", str(DB_PATH))
    if not Path(db_path).exists():
        st.warning(f"No database at {db_path}. Run scripts/run_experiments.py first.")
        return
    tables = load_tables(db_path)
    runs = tables["runs"]
    if runs.empty:
        st.warning("runs.db has no runs yet.")
        return
    present = ordered_policies(runs)
    picked = st.sidebar.multiselect("Policies", present, default=present,
                                    format_func=lambda p: POLICY_LABEL.get(p, p))

    tab_overview, tab_results, tab_policy = st.tabs(
        ["Overview", "Results", "Per-policy"])

    with tab_overview:
        st.subheader("Runs")
        cols = [c for c in ["run_id", "policy", "seed", "tag", "config_hash",
                            "git_commit", "start_time", "end_time"] if c in runs.columns]
        st.dataframe(runs[cols], use_container_width=True, hide_index=True)
        if "tag" in runs.columns and (runs.tag == "revision").any():
            st.caption("tag 'revision' = post-hoc reviewer revision, run after the step-5 "
                       "test results were seen; settings chosen on validation only.")
        st.caption(f"{len(runs)} runs · "
                   f"{runs.config_hash.nunique()} config hash(es) · "
                   f"{runs.git_commit.nunique()} git commit(s)")

    with tab_results:
        summary = summarize(tables)
        if summary.empty:
            st.info("No test metrics yet. Run src/evaluate.py to score the test days.")
        else:
            show_model_vs_chance(tables)
            show_gain_over_fixed(tables)
            show_bootstrap()
            show_learning_curve()
            st.subheader("Summary (mean ± std over seeds)")
            show = summary.copy()
            show.index = [POLICY_LABEL.get(p, p) for p in show.index]
            st.dataframe(show.round(4), use_container_width=True)

            for ref, bases in [("E", ["B", "C", "D", "E_minus_audit"]),
                               ("E_drift", ["E", "E_minus_audit", "E_minus_cluster",
                                            "E_drift_minus_cluster", "B", "C", "D"])]:
                if ref not in summary.index:
                    continue
                st.subheader(f"{POLICY_LABEL[ref]} vs others — high-vol macro-F1 "
                             "(percentage points)")
                diffs = []
                for base in bases:
                    if base in summary.index:
                        pp = (summary.loc[ref, "high_vol_F1_mean"]
                              - summary.loc[base, "high_vol_F1_mean"]) * 100
                        diffs.append({"comparison": f"{ref} − {POLICY_LABEL.get(base, base)}",
                                      "difference (pp)": round(pp, 3)})
                st.dataframe(pd.DataFrame(diffs), use_container_width=True, hide_index=True)

    with tab_policy:
        if not picked:
            st.info("Pick a policy in the sidebar.")
            return
        pol = st.selectbox("Policy", picked, format_func=lambda p: POLICY_LABEL.get(p, p))
        run_ids = runs[runs.policy == pol].run_id
        ups = tables["updates"]
        sels = tables["selections"]
        mets = tables["metrics"]

        c1, c2, c3 = st.columns(3)
        pol_ups = ups[ups.run_id.isin(run_ids)]
        c1.metric("runs (seeds)", runs[runs.policy == pol].seed.nunique())
        if not pol_ups.empty:
            c2.metric("updates accepted (avg)",
                      round((pol_ups.accepted == 1).groupby(pol_ups.run_id).sum().mean(), 1))
            c3.metric("rejected (avg)",
                      round((pol_ups.accepted == 0).groupby(pol_ups.run_id).sum().mean(), 1))

        if not pol_ups.empty:
            st.subheader("Validation macro-F1 during replay (mean over seeds)")
            traj = pol_ups.groupby("window").val_f1_after.mean().reset_index()
            start = pol_ups.sort_values("window").groupby("run_id").val_f1_before.first().mean()
            line = alt.Chart(traj).mark_line(color=MARK, strokeWidth=2).encode(
                x=alt.X("window:Q", title="stream window"),
                y=alt.Y("val_f1_after:Q", scale=alt.Scale(zero=False),
                        title="val macro-F1 (zoomed)"),
                tooltip=["window", alt.Tooltip("val_f1_after:Q", format=".4f")])
            base = alt.Chart(pd.DataFrame({"y": [start]})).mark_rule(
                color="#888", strokeDash=[4, 3]).encode(y="y:Q")
            st.altair_chart((base + line).properties(height=260), use_container_width=True)
            end = traj.val_f1_after.iloc[-1]
            st.caption(f"Dashed line = V1 before any update ({start:.4f}). After the last "
                       f"window: {end:.4f} ({(end - start) * 100:+.2f} pp). The y-axis is "
                       "zoomed so the change is visible; it spans well under one F1 point, "
                       "and on a 0–1 axis it would look flat.")

        pol_sel = sels[sels.run_id.isin(run_ids)]
        if not pol_sel.empty:
            st.subheader("Selection reasons")
            st.bar_chart(pol_sel.reason.value_counts().rename("count"))

        lat = mets[(mets.run_id.isin(run_ids)) & (mets.split == "latency")]
        if not lat.empty:
            p = lat.groupby("metric").value.mean()
            st.caption(f"Prediction latency per snapshot — p50 {p.get('p50_ms', float('nan')):.3f} ms, "
                       f"p99 {p.get('p99_ms', float('nan')):.3f} ms "
                       f"(model.predict_proba on one feature vector).")


if __name__ == "__main__":
    main()
