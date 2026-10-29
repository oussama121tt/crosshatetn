import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from crosshatetn.audit import run_audit
from crosshatetn.metrics import _macro_f1_batch, bootstrap_ci, paired_bootstrap, scores

LABELS = ["normal", "abusive", "hate"]


def labeled():
    return pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "source": "thsab",
        "label": ["normal", "hate", "normal", "hate"],
        "split": ["train", "train", "test", "test"],
        "group": ["a", "b", "c", "d"],
        "text_raw": ["un", "deux", "trois", "quatre"],
        "dedup_key": ["un", "deux", "trois", "quatre"],
    })


NO_PAIRS = pd.DataFrame({"id_a": [], "id_b": [], "sim": []})


def audit(df, pairs=NO_PAIRS):
    return run_audit(df, pairs, group_threshold=0.7, review_threshold=0.6, max_class_gap=1.0)


def test_audit_passes_on_clean_splits():
    assert audit(labeled()).ok


def test_audit_blocks_identical_text_across_splits():
    df = labeled()
    df.loc[2, "text_raw"] = "un"
    assert not audit(df).ok


def test_audit_blocks_shared_group_and_near_duplicates():
    df = labeled()
    df.loc[2, "group"] = "a"
    assert not audit(df).ok
    pairs = pd.DataFrame({"id_a": ["a"], "id_b": ["c"], "sim": [0.9]})
    assert not audit(labeled(), pairs).ok


def test_audit_lists_borderline_pairs_without_blocking():
    pairs = pd.DataFrame({"id_a": ["a"], "id_b": ["c"], "sim": [0.65]})
    result = audit(labeled(), pairs)
    assert result.ok and len(result.review_pairs) == 1


def test_scores_match_sklearn():
    y, p = ["normal", "hate", "abusive", "hate"], ["normal", "abusive", "abusive", "hate"]
    assert np.isclose(scores(y, p, LABELS)["macro_f1"], f1_score(y, p, labels=LABELS, average="macro"))


def test_fast_bootstrap_metric_matches_sklearn():
    rng = np.random.default_rng(0)
    yt, yp = rng.integers(0, 3, size=(5, 200)), rng.integers(0, 3, size=(5, 200))
    fast = _macro_f1_batch(yt, yp, 3)
    slow = [f1_score(t, p, labels=[0, 1, 2], average="macro", zero_division=0) for t, p in zip(yt, yp)]
    assert np.allclose(fast, slow)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.choice(LABELS, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.choice(LABELS, 300))
    point, low, high = bootstrap_ci(y, pred, LABELS, n_boot=500, seed=0)
    assert low <= point <= high


def test_paired_bootstrap_on_identical_systems():
    y = ["normal", "hate", "abusive"] * 20
    res = paired_bootstrap(y, y, y, LABELS, n_boot=200, seed=0)
    assert res["diff_macro_f1"] == 0 and res["p_value"] == 1.0
