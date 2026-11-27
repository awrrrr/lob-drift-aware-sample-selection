# Novelty statement (for the report)

*Post-hoc revision: written after the step-5 test results were seen.*

**What E_drift does.** Each update adds exactly B labelled samples from the newest window. An *audit* share is drawn at random from **confident** predictions (the lower-entropy half). The rest is *cluster-diverse uncertainty sampling*: shortlist the most uncertain, cluster them (PCA + k-means), take the most uncertain per cluster. The audit share is **set by a label-free drift score** d_k (window volatility relative to the training upper quartile): larger when d_k > 1.

**How it differs from prior work.**

- *Liu et al. (2023)* rank samples by uncertainty × local density (our baseline D adapts this). E_drift has no density term; it audits confident predictions and spreads uncertain picks across clusters.
- *Žliobaitė et al. (2011, 2014)* already argue that a drift-aware learner must "from time to time, label the instances about which the classifier is very certain", via a randomised uncertainty threshold (η ~ N(1, δ)) with fixed δ. **Auditing confident predictions is therefore their idea, not ours.** What differs: (i) the audit is an explicit share of a fixed per-update batch, not a per-instance threshold; (ii) audit draws come only from confident predictions; (iii) the share adapts to a measured, label-free drift score.
- *Zhdanov (2019)* pre-filters the most informative samples, clusters them, and picks one per cluster. Our cluster step is a close variant. **We do not claim it as new.**
- *Wang et al. (IJCAI 2026, NILE)*, the most recent related work, also splits a fixed labelling budget between random sampling and adaptive uncertainty sampling, and cites Liu et al. (2023) as a current hybrid strategy. E_drift follows the same random-plus-uncertainty recipe; it differs in selecting a per-window batch, drawing the random share only from confident predictions, and tying that share to measured drift.

**Contribution.** A drift-adaptive confident-prediction audit combined with cluster-diverse uncertainty sampling, evaluated under a fixed per-update budget on FI-2010 limit order book data in a leakage-safe replay (horizon-aware label availability, selection before label reveal, validation-gated updates, test scored once), with ablations isolating each component.

**Evidence.** Neither E nor E_drift beat random: 0.4149 vs 0.4151 high-volatility macro-F1 (bootstrap 95% CI for the difference −0.10 to +0.06 pp); no ablated component mattered beyond seed noise. The drift score finds volatile windows, but there the model is *more* confident and *not* less accurate. Only updating versus not updating mattered.

---
Sources checked: Zhdanov 2019 (arXiv:1901.05954, full text). Žliobaitė, Bifet, Holmes, Pfahringer 2011, *MOA Concept Drift Active Learning Strategies for Streaming Data*, JMLR W&CP 17 (full text). Wang et al. 2026, IJCAI-26 (full text). The 2014 TNNLS paper's Split strategy is described from secondary sources only. Further 2024–2026 references: `related_work.md`.
