"""Streamlit viewer for the experiment logs. Reads runs.db only.

This is a viewer, not the contribution: every number shown comes straight from
runs.db (written by the harness and src/evaluate.py). It never loads the raw
data, retrains, or recomputes a metric. Run with:

    .venv/bin/streamlit run app/dashboard.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "runs.db"

POLICY_ORDER = ["A", "B", "C", "D", "E", "E_minus_audit"]
POLICY_LABEL = {"A": "A fixed", "B": "B random", "C": "C uncertainty",
                "D": "D density×unc", "E": "E proposed", "E_minus_audit": "E−audit"}


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


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #

def main():
    st.set_page_config(page_title="LOB drift-aware selection — runs viewer", layout="wide")
    st.title("LOB drift-aware sample selection — experiment viewer")
    st.caption("A viewer over runs.db. Every number is read from the logs — "
               "nothing here retrains or recomputes.")

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
        st.dataframe(runs[["run_id", "policy", "seed", "config_hash",
                           "git_commit", "start_time", "end_time"]],
                     use_container_width=True, hide_index=True)
        st.caption(f"{len(runs)} runs · "
                   f"{runs.config_hash.nunique()} config hash(es) · "
                   f"{runs.git_commit.nunique()} git commit(s)")

    with tab_results:
        summary = summarize(tables)
        if summary.empty:
            st.info("No test metrics yet. Run src/evaluate.py to score the test days.")
        else:
            st.subheader("High-volatility test macro-F1 (primary)")
            hv = summary[["high_vol_F1_mean"]].rename(
                columns={"high_vol_F1_mean": "high-vol macro-F1"})
            hv.index = [POLICY_LABEL.get(p, p) for p in hv.index]
            st.bar_chart(hv)
            st.subheader("Summary (mean ± std over seeds)")
            show = summary.copy()
            show.index = [POLICY_LABEL.get(p, p) for p in show.index]
            st.dataframe(show.round(4), use_container_width=True)

            if "E" in summary.index:
                st.subheader("E vs baselines — high-vol macro-F1 (percentage points)")
                diffs = []
                for base in ["B", "C", "D", "E_minus_audit"]:
                    if base in summary.index:
                        pp = (summary.loc["E", "high_vol_F1_mean"]
                              - summary.loc[base, "high_vol_F1_mean"]) * 100
                        diffs.append({"comparison": f"E − {POLICY_LABEL.get(base, base)}",
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
            traj = pol_ups.groupby("window").val_f1_after.mean()
            st.line_chart(traj.rename("val macro-F1"))

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
