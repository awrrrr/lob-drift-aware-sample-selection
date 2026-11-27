# Related work: recent references (2024–2026)

*Added after the faculty review comment that the base paper (Liu et al., 2023) is not recent enough.
Nothing in the experiments changed; these references position the work, they do not alter it.*

## Why Liu et al. (2023) stays cited

Policy D is adapted from Liu et al.'s uncertainty × density selection, so it must stay cited. The
2024–2026 papers below show that it is still treated as a current approach and that the field has
moved toward the same random-plus-uncertainty mix that policy E uses.

## Closest recent work

**Wang, H., Liu, H., Hao, Q., Koh, Y. S., Yang, Q., & Shao, J. (2026). From neural collapse to
label-limited evolving streams: Geometry-constrained learning under dynamic class imbalance.
*Proceedings of IJCAI-26*, 3070–3078.** (NILE; full text read. Code: github.com/WangHLzZ/NILE)

- Setting: a drifting data stream where only a share of samples can be labelled (budget B = 0.2,
  decided per instance).
- Selection ("hybrid active learning"), run in order for each arriving sample:
  1. random sampling with probability B/2, "to ensure basic coverage of the current data stream
     distribution";
  2. uncertainty sampling, with a threshold that adapts from quantiles of recent correct and
     incorrect predictions;
  3. sampling aimed at minority classes.
- Cites Liu et al. (2023) as the "hybrid strategies" line of active learning for streams.
- Baselines: CALMID, MicFoal and LEPID (active learning), plus supervised and semi-supervised
  stream learners. General stream benchmarks, reported mainly as G-mean; no order-book data.

**Relation to this project.** Policy E has the same structure: a random share for coverage plus
uncertainty-based picks with a diversity step. E differs in three ways: it selects a batch of B per
window rather than deciding per instance; its random share is drawn only from confident
predictions (an audit of the model's confident calls); and in E_drift that share follows a
label-free drift score. NILE is a deep, class-imbalance method; this project uses linear and kNN
models on FI-2010 under a leakage-safe replay.

## Supporting references

- **Cacciarelli, D., & Kulahci, M. (2024). Active learning for data streams: a survey.
  *Machine Learning*, 113(1), 185–239.** doi:10.1007/s10994-023-06454-2. The standard overview of
  stream-based active learning; use it in place of older surveys. (Springer record and NILE's
  reference list.)
- **Jiao, B., Gomes, H. M., Xue, B., Guo, Y., & Zhang, M. (2025). A semi-supervised active learning
  neural network for data streams with concept drift. *IEEE Computational Intelligence Magazine*,
  20(1), 18–33.** Active learning under drift with a labelling budget; queries samples by a
  smoothness loss and also learns from unlabelled data. (Abstract and NILE's reference list only;
  full text not read.)
- **Prata, M., Masi, G., Berti, L., Arrigoni, V., Coletta, A., Cannistraci, I., Vyetrenko, S.,
  Velardi, P., & Bartolini, N. (2024). LOB-based deep learning models for stock price trend
  prediction: a benchmark study. *Artificial Intelligence Review*, 57(5).**
  doi:10.1007/s10462-024-10715-4. Order-book domain reference: re-runs 15 deep models on FI-2010 and
  finds most overfit it and drop sharply on unseen market data (LOBSTER). Relevant to the choice of
  simple models. (Abstract and preprint excerpts; full published text not read.)

## Viva lines

- "Is your base paper outdated?" — "Liu et al. 2023 is still cited as a current hybrid strategy
  by NILE at IJCAI 2026. NILE itself uses the same recipe as my policy E: a random share plus
  uncertainty sampling under a fixed budget. Mine is a window-based version tested on order-book
  data under a strict no-leakage protocol."
- Be honest about the gap: NILE decides per sample with deep features and targets class
  imbalance; this project selects batches and targets volatility drift.

*Check before quoting: NILE's entry is from its own PDF; Prata et al.'s authors are from the arXiv
page (arXiv:2308.01915) and the volume from the Springer listing; Jiao et al.'s entry is from
NILE's reference list. Confirm the last two against the publisher pages before final submission.*
