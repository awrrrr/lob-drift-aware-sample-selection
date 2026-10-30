"""Sample-selection policies.

Every policy has the same signature and contract:

    policy(pool, proba, B, rng, selected_so_far, cfg) -> list[(sample_id, reason)]

    pool            DataFrame of the window's selectable samples: a sample_id
                    column plus the feature columns. Labels are NOT included.
    proba           (len(pool), 3) array of the current model's class
                    probabilities for those rows, in the same order as `pool`.
    B               budget: the number of samples to return.
    rng             numpy Generator for any randomness (seeded by the run).
    selected_so_far set of sample_ids already selected in earlier windows.
    cfg             the loaded config (policies: block holds the knobs).

    Returns at most B rows, each a (sample_id, reason) pair, with unique ids
    drawn only from `pool`. Fewer than B only when the pool itself is smaller.

Policies never see labels (leakage rule 2). Uncertainty is predictive entropy
everywhere. C and D are deterministic; B and E use `rng`.
"""

import numpy as np

EPS = 1e-12


def entropy(proba):
    """Predictive entropy per row (natural log); higher = more uncertain."""
    p = np.clip(proba, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _ids_feats(pool):
    feats = [c for c in pool.columns if c != "sample_id"]
    return pool["sample_id"].to_numpy(), pool[feats].to_numpy(), feats


def _take_all(ids, tag):
    return [(int(s), tag) for s in ids]


def _top_b(ids, score, B, tag):
    """The B rows with the highest score; stable, so ties break by pool order."""
    order = np.argsort(-score, kind="stable")[:B]
    return [(int(ids[i]), tag) for i in order]


# --------------------------------------------------------------------------- #
# A, B: baselines (also in step 3)
# --------------------------------------------------------------------------- #

def fixed_policy(pool, proba, B, rng, selected_so_far, cfg):
    """Policy A: never select anything; the model stays V1 for the whole run."""
    return []


def random_policy(pool, proba, B, rng, selected_so_far, cfg):
    """Policy B: B samples drawn uniformly at random from the pool."""
    ids = pool["sample_id"].to_numpy()
    k = min(B, len(ids))
    return [(int(s), "random") for s in rng.choice(ids, size=k, replace=False)]


# --------------------------------------------------------------------------- #
# C: uncertainty
# --------------------------------------------------------------------------- #

def uncertainty_policy(pool, proba, B, rng, selected_so_far, cfg):
    """Policy C: the B most uncertain samples by predictive entropy."""
    ids = pool["sample_id"].to_numpy()
    if len(ids) <= B:
        return _take_all(ids, "uncertainty")
    return _top_b(ids, entropy(proba), B, "uncertainty")


# --------------------------------------------------------------------------- #
# D: density x uncertainty (adapted from Liu et al. 2023)
# --------------------------------------------------------------------------- #

def density_uncertainty_policy(pool, proba, B, rng, selected_so_far, cfg):
    """Policy D: top-B by entropy x local density, computed within the window.

    Local density = 1 / (1 + mean distance to the k nearest neighbours in
    feature space, among the pool). This is a window-based adaptation of the
    uncertainty x density criterion in Liu et al. 2023; it is not their CogDQS.
    """
    from sklearn.neighbors import NearestNeighbors

    ids, X, _ = _ids_feats(pool)
    if len(ids) <= B:
        return _take_all(ids, "density_uncertainty")
    k = min(cfg["policies"]["density_k"], len(ids) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)   # +1: the point itself
    dist, _ = nn.kneighbors(X)
    density = 1.0 / (1.0 + dist[:, 1:].mean(axis=1))   # drop self-distance (0)
    return _top_b(ids, entropy(proba) * density, B, "density_uncertainty")


# --------------------------------------------------------------------------- #
# E: proposed (audit + uncertainty clustering); E-audit sets audit_share = 0
# --------------------------------------------------------------------------- #

def proposed_policy(pool, proba, B, rng, selected_so_far, cfg, audit_share=None):
    """Policy E.

    Reserve an audit share of B for random samples drawn from confident
    (low-entropy) predictions, to keep checking where the model is sure. Fill
    the rest from a shortlist of the most uncertain samples: reduce with PCA,
    cluster with k-means, and take the most uncertain sample per cluster, so the
    uncertain picks are spread across feature-space regions rather than bunched.

    Fallbacks: pool <= B -> take all; too few confident samples -> the unused
    audit slots go to the cluster step; shortlist not larger than the cluster
    budget -> take the most uncertain directly; empty clusters -> fill with the
    next most uncertain. No repeated ids.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    pcfg = cfg["policies"]
    if audit_share is None:
        audit_share = pcfg["audit_share"]
    ids, X, _ = _ids_feats(pool)
    n = len(ids)
    if n <= B:
        return _take_all(ids, "proposed:all")

    H = entropy(proba)
    km_seed = int(rng.integers(0, 2**31 - 1))
    n_audit = int(round(audit_share * B))
    chosen, chosen_idx = [], set()

    # --- audit: random draws from the most confident (lowest-entropy) samples ---
    if n_audit > 0:
        thr = np.quantile(H, pcfg["confident_quantile"])
        confident = np.where(H <= thr)[0]
        if len(confident) == 0:
            confident = np.argsort(H, kind="stable")[: max(n_audit, 1)]
        take = min(n_audit, len(confident))
        for i in rng.choice(confident, size=take, replace=False):
            chosen.append((int(ids[i]), "audit"))
            chosen_idx.add(int(i))

    n_cluster = B - len(chosen)

    # --- cluster step: most-uncertain shortlist, spread across k-means clusters ---
    remaining = np.array([i for i in range(n) if i not in chosen_idx])
    order = remaining[np.argsort(-H[remaining], kind="stable")]       # uncertain first
    short = order[: min(pcfg["shortlist_factor"] * n_cluster, len(order))]

    if n_cluster <= 0:
        pass
    elif len(short) <= n_cluster:
        for i in short[:n_cluster]:
            chosen.append((int(ids[i]), "cluster:direct"))
            chosen_idx.add(int(i))
    else:
        ncomp = min(pcfg["pca_components"], len(short), X.shape[1])
        Z = PCA(n_components=ncomp, random_state=km_seed).fit_transform(X[short])
        labels = KMeans(n_clusters=n_cluster, n_init=10, random_state=km_seed).fit_predict(Z)
        for c in range(n_cluster):
            members = short[labels == c]
            if len(members):
                best = members[np.argmax(H[members])]
                chosen.append((int(ids[best]), "cluster"))
                chosen_idx.add(int(best))
        if len(chosen) < B:   # empty clusters -> fill with next most uncertain
            for i in order:
                if int(i) not in chosen_idx:
                    chosen.append((int(ids[i]), "cluster:fill"))
                    chosen_idx.add(int(i))
                    if len(chosen) >= B:
                        break
    return chosen[:B]


def proposed_no_audit_policy(pool, proba, B, rng, selected_so_far, cfg):
    """Ablation E-audit: policy E with the audit share given to the cluster step."""
    return proposed_policy(pool, proba, B, rng, selected_so_far, cfg, audit_share=0.0)


POLICIES = {
    "A": fixed_policy,
    "B": random_policy,
    "C": uncertainty_policy,
    "D": density_uncertainty_policy,
    "E": proposed_policy,
    "E_minus_audit": proposed_no_audit_policy,
}
