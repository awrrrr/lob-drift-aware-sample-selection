"""Sample-selection policies.

Every policy has the same signature and contract:

    policy(pool, proba, B, rng, selected_so_far) -> list[(sample_id, reason)]

    pool            DataFrame of the window's selectable samples (features +
                    sample_id), labels NOT included.
    proba           (len(pool), 3) array of the current model's class
                    probabilities for those rows, in the same order as `pool`.
    B               budget: the number of samples to return.
    rng             numpy Generator for any randomness (seeded by the run).
    selected_so_far set of sample_ids already selected in earlier windows.

    Returns at most B rows, each a (sample_id, reason) pair, with unique ids
    drawn only from `pool`. Fewer than B only when the pool itself is smaller.

Policies never see labels (leakage rule 2). Step 3 ships A (fixed) and B
(random); C, D and E are added in step 4.
"""


def fixed_policy(pool, proba, B, rng, selected_so_far):
    """Policy A: never select anything; the model stays V1 for the whole run."""
    return []


def random_policy(pool, proba, B, rng, selected_so_far):
    """Policy B: B samples drawn uniformly at random from the pool."""
    ids = pool["sample_id"].to_numpy()
    k = min(B, len(ids))
    chosen = rng.choice(ids, size=k, replace=False)
    return [(int(s), "random") for s in chosen]


POLICIES = {
    "A": fixed_policy,
    "B": random_policy,
}
