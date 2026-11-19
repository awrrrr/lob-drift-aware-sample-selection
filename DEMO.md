# Demo guide — LOB Drift-Aware Sample Selection

A 5-minute walkthrough for the viva. Read `CLAUDE.md` first for the full protocol;
this file is just how to **start it** and **what to show**.

## The one-line pitch

> We study *which* new limit-order-book samples a model should learn from when each
> update can only add a fixed budget of labelled samples. We built a leakage-safe
> replay harness that compares six selection policies fairly on FI-2010, and we
> report honestly what it found — including that our proposed policy does **not**
> beat random at this budget, and *why*.

The contribution is the **policy + the fair test**, not the dashboard. The dashboard
is only a viewer over `runs.db`.

---

## 0. Prerequisites (already set up on this machine)

- Python venv at `.venv/` (pandas, numpy, scikit-learn, streamlit, matplotlib, pyyaml, pytest).
- Raw FI-2010 files in `data/raw/` (NoAuction + ZScore only). **Gitignored** — not on GitHub.
- `data/processed/fi2010.pkl`, `runs.db`, and `results/` already built. **Also gitignored.**

If you cloned this fresh from GitHub, you only have the code — you'd need the FI-2010
data and a full pipeline run (section 4) before the dashboard shows anything.

---

## 1. Fast demo — launch the dashboard (recommended)

Everything is already computed, so just view it:

```bash
.venv/bin/python -m streamlit run app/dashboard.py --server.port 8520
```

Open **http://localhost:8520**.

> Use the `-m streamlit` form, not `.venv/bin/streamlit` — the folder path has a
> space in it, which breaks the console-script wrapper.

Stop it when done:

```bash
pkill -f "streamlit run app/dashboard.py"
```

---

## 2. What to show — tab by tab

### Overview tab — "every number is traceable"
Point at: **18 runs · 1 config hash · 1 git commit.**
> "Every result comes from one frozen config and one commit. Nothing is hand-entered;
> the dashboard reads straight from `runs.db`, which the harness wrote."

The 18 runs are: A (1 seed, no randomness), C and D (1 seed each — deterministic
top-B rules), and B, E, E−audit (5 seeds each — they use randomness, so we measure
spread). That's 1+1+1+5+5+5 = 18.

### Results tab — the headline (and the honest finding)
The bar chart is **high-volatility test macro-F1 by policy**. All six bars sit at
≈ 0.415.
> "This flatness *is* the result. Once you update at all, the smart policies (C, D, E)
> don't beat random (B) within seed noise. The only real gain is updating vs. the
> fixed model — about +0.25 points — and random gets that too."

Scroll down for the **summary table** (overall / low-vol / accuracy, updates
accepted vs rejected, latency) and the **E vs baselines** differences in percentage
points. Note `E − E−audit = −0.009 pp`: the audit share didn't help here.

### Per-policy tab — the mechanism
Pick a policy to show updates accepted/rejected, the validation-F1 trajectory during
replay, the selection-reason breakdown, and per-snapshot prediction latency (p50/p99).
> "The policies really do pick different samples — D selects 61% Stationary, and
> cross-policy overlap is low (Jaccard ≤ 0.35). They just don't *matter* at this budget."

---

## 3. Defending the negative result (likely viva question)

**"So your contribution failed?"**
> "No. A result only counts if the setup *could* have shown a win and didn't. We built
> a harness strict enough to detect one — same V1, same stream, same budget, a
> validation gate, 5 seeds — and it honestly reports no separation. We then diagnosed
> *why* instead of tuning to force a win (which would have broken the protocol)."

The diagnosis (`scripts/failure_analysis.py`, `results/learning_curve.png`):
- Budget B=50/window is only **2.6%** of the 178k training base — it lands on the
  flat early part of the learning curve where smart ≈ random.
- The model doesn't saturate: adding the whole stream pushes overall F1 to 0.4432.
- So selection would only matter with a **smaller base or larger budget** — stated as
  **future work**, not re-run, because changing the setup after seeing test results
  would break leakage rule 3.

**"How do you know there's no data leakage?"**
Point to `tests/` (budget + leakage tests), and the four rules in `CLAUDE.md`:
labels usable only after the horizon passes, selection never reads labels first, test
days scored once by `scripts/evaluate.py` with no settings changed in response, and
the volatility threshold frozen on training days only.

---

## 4. Optional — run the pipeline from scratch (shows it end-to-end)

Steps 1–3 are fast (seconds–minutes); step 4 is the slow one (the policy × seed sweep).
Run in this order:

```bash
.venv/bin/python scripts/check_data.py
```
```bash
.venv/bin/python scripts/train_v1.py
```
```bash
.venv/bin/python scripts/freeze_volatility.py
```
```bash
.venv/bin/python scripts/run_experiments.py --policies A B C D E E_minus_audit --seeds 0 1 2 3 4 --skip-existing
```
```bash
.venv/bin/python scripts/evaluate.py
```
```bash
.venv/bin/python scripts/make_tables.py
```

`--skip-existing` makes step 4 a no-op for runs already in `runs.db`, so it's safe to
re-run live. `evaluate.py` scores the test days deterministically and reproduces the
same numbers every time.

Run the tests to show the leakage/budget guarantees pass:

```bash
.venv/bin/python -m pytest -q
```

---

## 5. Quick reference

| Thing | Where |
|---|---|
| Frozen protocol & all settings | `CLAUDE.md`, `config.yaml` |
| Results tables & plots | `results/summary.csv`, `results/differences.csv`, `results/*.png` |
| Raw numbers for every run | `runs.db` (tables: runs, selections, updates, metrics) |
| Policies (A–E, ablation) | `src/policies.py` |
| Replay harness + gate | `src/harness.py` |
| Dashboard (viewer only) | `app/dashboard.py` |
