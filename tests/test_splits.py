import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from crosshatetn.splits import (SplitError, assign, check_frozen_digests, fresh_assignment, hashed_split,
                                load_frozen, load_manifest, save_frozen)

CFG = {"test": 0.15, "validation": 0.15, "n_folds": 20, "stratify_on": ["source", "label"]}


def units(n=600, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["normal", "abusive", "hate"], size=n, p=[0.6, 0.25, 0.15])
    sources = rng.choice(["thsab", "lhsab"], size=n)
    ids = [f"{s}-{i:05d}" for i, s in enumerate(sources)]
    groups = [ids[i - 1] if i % 7 == 0 and i > 0 else ids[i] for i in range(n)]  # quelques paires groupées
    return pd.DataFrame({"id": ids, "source": sources, "label": labels, "group": groups})


def test_fresh_assignment_keeps_groups_together_and_hits_proportions():
    u = units()
    split = fresh_assignment(u, CFG, seed=42)
    df = u.assign(split=split.to_numpy())
    assert (df.groupby("group")["split"].nunique() == 1).all()
    shares = df["split"].value_counts(normalize=True)
    assert abs(shares["test"] - 0.15) < 0.03 and abs(shares["validation"] - 0.15) < 0.03


def test_fresh_assignment_is_deterministic():
    u = units()
    assert fresh_assignment(u, CFG, 42).equals(fresh_assignment(u, CFG, 42))


def test_frozen_assignments_never_change_and_new_ids_inherit_their_group():
    u = units()
    frozen = fresh_assignment(u, CFG, 42)
    grown = pd.concat([u, pd.DataFrame({"id": ["new-1", "new-2"], "source": ["thsab", "thsab"],
                                        "label": ["hate", "normal"], "group": [u.loc[0, "group"], "new-2"]})],
                      ignore_index=True)
    outcome = assign(grown, frozen, CFG, seed=42)
    assert outcome.assignment.loc[frozen.index].equals(frozen)
    assert outcome.assignment["new-1"] == frozen[u.loc[0, "id"]]          # hérite du split de son groupe
    assert outcome.assignment["new-2"] == hashed_split("new-2", CFG, 42)  # nouveau groupe : placement haché
    assert set(outcome.added["origine"]) == {"groupe", "hachage"}


def test_group_split_across_frozen_splits_is_an_error():
    u = units(50)
    frozen = pd.Series(["train"] * 50, index=u["id"])
    frozen.iloc[1] = "test"
    u["group"] = "g"  # tout le monde dans le même groupe, mais deux splits gelés différents
    try:
        assign(u, frozen, CFG, seed=42)
    except SplitError:
        return
    raise AssertionError("un groupe réparti sur plusieurs splits doit lever SplitError")


def test_saved_assignment_is_checked_against_its_manifest():
    u = units()
    outcome = assign(u, None, CFG, seed=42)
    with tempfile.TemporaryDirectory() as tmp:
        save_frozen(Path(tmp), outcome, None, u, {"graine": 42})
        frozen, manifest = load_frozen(Path(tmp)), load_manifest(Path(tmp))
        check_frozen_digests(frozen, manifest)                          # intact : pas d'erreur
        assert manifest["historique"][0]["action"] == "création"
        tampered = frozen.copy()
        tampered[tampered.index[tampered == "test"][0]] = "train"       # modification « à la main »
        try:
            check_frozen_digests(tampered, manifest)
        except SplitError:
            return
    raise AssertionError("un test modifié hors du pipeline doit être détecté")
