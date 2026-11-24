# Novelty statement (for the report)

*Post-hoc revision: written after the step-5 test results were seen.*

**What E_drift does.** Each update adds exactly B labelled samples from the newest window. An *audit* share is drawn at random from **confident** predictions (the lower-entropy half). The rest is *cluster-diverse uncertainty sampling*: shortlist the most uncertain, cluster them (PCA + k-means), take the most uncertain per cluster. The audit share is **set by a label-free drift score** d_k (window volatility relative to the training upper quartile): larger when d_k > 1. B never changes.

**How it differs from prior work.**

- *Liu et al. (2023)* rank samples by uncertainty × local density (our baseline D adapts this). E_drift has no density term; it audits confident predictions and spreads uncertain picks across clusters.
- *Žliobaitė et al. (2011, 2014)* already argue that a drift-aware learner must "from time to time, label the instances about which the classifier is very certain". They do this by randomising the uncertainty threshold for each instance (η ~ N(1, δ)), with δ fixed. Their 2014 Split strategy additionally labels part of the stream at random. **The idea of auditing confident predictions is therefore theirs, not ours.** What differs here: (i) the audit is an explicit share of a fixed per-update batch budget, not a per-instance threshold; (ii) audit draws come only from confident predictions, not from the whole stream; (iii) the share adapts to a measured, label-free drift score instead of being fixed.
- *Zhdanov (2019)* pre-filters the βk most informative samples, clusters them with weighted k-means, and picks the sample nearest each centre. Our cluster step is a close variant (most uncertain per cluster, unweighted k-means on PCA features). Zhdanov reports that variant performing the same. **We do not claim the cluster step as new.**

**Contribution.** A drift-adaptive confident-prediction audit combined with cluster-diverse uncertainty sampling, evaluated under a fixed per-update budget on FI-2010 limit order book data in a leakage-safe replay protocol (horizon-aware label availability, selection before label reveal, validation-gated updates, test scored once), with ablations isolating each component.

**Evidence.** Neither E nor E_drift beat random: E_drift scored 0.4149 high-volatility macro-F1 vs 0.4151 for random (bootstrap 95% CI for the difference −0.10 to +0.06 pp), and no ablated component changed results beyond seed noise. The drift score finds volatile windows, but there the model is *more* confident and *not* less accurate. Only updating versus not updating mattered.

---
Sources checked: Zhdanov 2019 (arXiv:1901.05954, full text: pre-filter βk, weighted k-means, nearest-to-centre pick). Žliobaitė, Bifet, Holmes, Pfahringer 2011, *MOA Concept Drift Active Learning Strategies for Streaming Data*, JMLR W&CP 17 (full text: randomised-threshold quote, Algorithm 5, DDM-based classifier replacement). The 2014 TNNLS paper's Split strategy is described from secondary sources only; its full text was not accessible.
