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
Read it top to bottom; it answers four questions in order. Separate them out loud:
"the model learns" and "selection doesn't matter" are different claims.

1. **Did the model learn?** V1 scores 0.4335 macro-F1 against 0.333 for random
   guessing (+10 pp). Every class is above the orange chance line; always predicting
   one class scores only ≈ 0.17.
   > "So the model clearly learns. The flat results below are about *which samples*
   > it learns from, not whether it learns."
2. **What does updating add?** Each dot is one seed's gain over the fixed model A, in
   percentage points. On high-volatility windows every policy sits right of zero
   (about +0.2 to +0.3 pp) and the policies overlap. Switch to "all" or "low-volatility"
   to show the gains hug zero there.
3. **Is any difference significant?** E_drift vs each policy with bootstrap 95% CIs.
   Only the interval against A clears zero.
4. **Why so flat?** The learning curve: more data does help (0.4335 → 0.4432 with all
   93k stream samples), but the budget used (4,650 samples, the dashed line) adds only
   2.6% to the 178k base.
   > "This flatness *is* the result. Once you update at all, the smart policies don't
   > beat random within seed noise, because the budget is too small next to the base."

The axes are honest: bars start at zero, and the gain/CI charts are in percentage
points because the differences themselves are fractions of a point. If asked, say so.

Scroll down for the **summary table** (overall / low-vol / accuracy, updates
accepted vs rejected, latency) and the **E vs baselines** differences in percentage
points. Note `E − E−audit = −0.009 pp`: the audit share didn't help here.

### Per-policy tab — the mechanism
Pick a policy to show updates accepted/rejected, the validation-F1 trajectory during
replay (zoomed axis, dashed line = V1 before any update; the caption gives the change), the selection-reason breakdown, and per-snapshot prediction latency (p50/p99).
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

## 5. Revisions after the faculty review (post-hoc)

The review asked to (i) define how drift is detected and (ii) show how selection
changes under drift. Everything here was added **after** the step-5 test results
were seen; settings were chosen on day-6 validation only and the test days were
scored once. Full record: `results/revision/LOG.md`.

**The drift indicator (d_k).** For each update window: the average recent mid-price
volatility of the window's samples ÷ the volatility threshold frozen on training
days. d_k > 1 means "more volatile than the top quarter of training windows". It
uses only past prices, never labels (`src/data.py: drift_score`).
Show `results/revision/drift_trajectory.png`. 13 of 93 stream windows have d_k > 1.

**How selection changes under drift (E_drift).** Same budget of 50 per window. In
calm windows 10% of it (5 picks) audits confident predictions and the rest goes to
cluster-diverse uncertain samples. When d_k > 1, the audit rises to 60% (30 picks).
The 0.1 / 0.6 split was picked on validation (`audit_grid.csv`).

**What it found** (`ablation.csv`, `bootstrap.csv`):
- E_drift: 0.4149 high-vol macro-F1, vs 0.4151 for random. **It does not win**; the
  95% CI for the difference is −0.10 to +0.06 pp.
- Removing the audit, the clustering, or the drift adaptation changes nothing beyond
  seed noise. Only updating vs not updating matters (+0.23 pp, CI clear of zero).
- Why (`drift_vs_v1.png`): volatile windows are where the model is *more* confident and
  *not* more wrong, so auditing more there has little to correct.

**Are the new results significant?** No, except one weak comparison. Paired bootstrap
over test windows (2000 resamples, seed-averaged), high-vol macro-F1, in pp
(`results/revision/bootstrap.csv`):

| E_drift vs | Diff (pp) | 95% CI (pp) | Significant? |
|---|---|---|---|
| B random | −0.015 | −0.10 to +0.06 | No |
| C uncertainty | −0.010 | −0.20 to +0.16 | No |
| D density×uncertainty | −0.018 | −0.21 to +0.19 | No |
| E (fixed audit) | −0.040 | −0.10 to +0.02 | No |
| E−audit | −0.049 | −0.11 to +0.02 | No |
| E−cluster | −0.062 | −0.17 to +0.02 | No |
| E_drift−cluster | +0.030 | −0.11 to +0.14 | No |
| A fixed (no updates) | +0.234 | +0.03 to +0.44 | Yes, barely |

- Drift adaptation, the audit and clustering have no measurable effect: the differences
  are hundredths of a point, the same size as the seed-to-seed spread.
- The one significant result (vs A) only shows that updating at all helps, and random
  gets the same gain. It is small (0.23 pp), rests on just 9 high-vol test windows,
  vanishes over all 85 test windows (every all-window CI contains 0), and sits at the
  edge of a Bonferroni correction for the 8 comparisons.
- "Not significant" is not "no effect": with 9 high-vol windows the intervals are
  ±0.1–0.2 pp wide. The honest claim is that **any effect of the selection strategy is
  smaller than ~0.1–0.2 F1 points at this budget**, which is practically negligible.

> Viva answer: "No. The only significant result is updating vs the fixed model, and
> random gets that too. The intervals rule out any practically meaningful gain from
> smart selection at B = 50, which matches the learning-curve diagnosis."

**Likely question: "So the drift-aware version also failed?"**
> "Yes, and the drift diagnostic explains why. Volatility does detect a regime change,
> but on this stream the change makes the model more confident and no less accurate.
> So spending more of the budget auditing confident predictions there can't help. I
> picked the settings on validation, scored test once, and report the null result
> with confidence intervals instead of tuning until something won."

**Novelty, honestly** (`results/revision/novelty.md`): auditing confident predictions
to catch drift goes back to Žliobaitė et al.; the cluster step is close to Zhdanov
(2019). What's ours is the explicit, drift-adaptive audit share under a fixed
per-update budget, tested in a leakage-safe replay. Leakage rules → tests:
`results/revision/leakage.md`.

**In the dashboard:** E_drift and the two −cluster ablations appear as "(revision)"
policies, and the Results tab adds an "E_drift vs others" table. Grid and check runs
are hidden.

**Not done (future work):** budget sweep (B = 25…500) and a second independent
dataset (no comparable free labelled LOB dataset; LOBSTER is paid).

---

## 6. Quick reference

| Thing | Where |
|---|---|
| Frozen protocol & all settings | `CLAUDE.md`, `config.yaml` |
| Results tables & plots | `results/summary.csv`, `results/differences.csv`, `results/*.png` |
| Revision results (post-hoc) | `results/revision/` (LOG.md first) |
| Raw numbers for every run | `runs.db` (tables: runs, selections, updates, metrics, timings) |
| Policies (A–E, ablation, E_drift) | `src/policies.py` |
| Drift score | `src/data.py: drift_score` |
| Replay harness + gate | `src/harness.py` |
| Dashboard (viewer only) | `app/dashboard.py` |
