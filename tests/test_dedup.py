import tempfile
from pathlib import Path

import pandas as pd

from crosshatetn.dedup import (connected_groups, mark_duplicates, near_duplicate_groups, read_arbitration,
                               resolve_conflicts, similar_pairs)
from crosshatetn.text import add_text_features

LABELS = ["normal", "abusive", "hate"]
NO_ARBITRATION = pd.DataFrame(columns=["id_representant", "label_arbitre", "arbitre", "date", "commentaire"])


def corpus(rows):
    df = pd.DataFrame(rows, columns=["text_raw", "label"])
    df["id"] = [f"src-{i:05d}" for i in range(1, len(df) + 1)]
    df["source"], df["role"], df["source_row"] = "src", "labeled", range(1, len(df) + 1)
    return add_text_features(df)


def test_duplicates_keep_first_occurrence_and_flag_empty_texts():
    df = mark_duplicates(corpus([("تونس !!", "normal"), ("  ", "normal"), ("تونس", "normal"), ("autre", "hate")]))
    assert df["status"].tolist() == ["ok", "vide", "doublon", "ok"]
    assert df.loc[2, "dup_group"] == "src-00001"


def test_conflicts_are_excluded_by_default():
    df = mark_duplicates(corpus([("جاهل", "normal"), ("جاهل", "abusive"), ("سلام", "normal")]))
    out, conflicts = resolve_conflicts(df, NO_ARBITRATION, "exclude")
    assert out["status"].tolist() == ["conflit", "doublon", "ok"]
    assert conflicts["resolution"].tolist() == ["exclu"]


def test_majority_policy_needs_strict_majority():
    df = mark_duplicates(corpus([("لعنة", "abusive"), ("لعنة", "abusive"), ("لعنة", "normal"),
                                 ("طز", "normal"), ("طز", "abusive")]))
    out, conflicts = resolve_conflicts(df, NO_ARBITRATION, "majority")
    assert out.loc[0, "label"] == "abusive" and out.loc[0, "label_status"] == "majorite"
    assert out.loc[3, "status"] == "conflit"  # égalité : pas de majorité stricte


def test_arbitration_overrides_policy():
    df = mark_duplicates(corpus([("جاهل", "normal"), ("جاهل", "abusive")]))
    arbitration = pd.DataFrame([{"id_representant": "src-00001", "label_arbitre": "abusive", "arbitre": "A",
                                 "date": "", "commentaire": ""}])
    out, conflicts = resolve_conflicts(df, arbitration, "exclude")
    assert out.loc[0, "status"] == "ok"
    assert (out.loc[0, "label"], out.loc[0, "label_status"]) == ("abusive", "arbitre")


def test_read_arbitration_validates_labels():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "arbitrage.csv"
        path.write_text("id_representant,label_arbitre,arbitre,date,commentaire\nsrc-00001,toxique,A,,\n",
                        encoding="utf-8")
        try:
            read_arbitration(path, LABELS)
        except ValueError:
            return
    raise AssertionError("une étiquette inconnue doit être refusée")


def test_similar_pairs_finds_near_duplicates_only():
    pairs = similar_pairs(["التوانسة فسااد كلهم", "شك التوانسة فسااد كلهم", "صباح الخير يا جماعة"], (3, 5), 0.7)
    assert pairs[["i", "j"]].values.tolist() == [[0, 1]]


def test_connected_groups_follow_chains():
    pairs = pd.DataFrame({"i": [0, 1], "j": [1, 2], "sim": [0.9, 0.8]})
    assert connected_groups(4, pairs).tolist() == [0, 0, 0, 3]


def test_group_id_is_smallest_member_id():
    units = pd.DataFrame({"id": ["b-1", "a-1", "c-1"],
                          "text_light": ["شك التوانسة فسااد كلهم", "التوانسة فسااد كلهم", "صباح الخير يا جماعة"]})
    group, pairs = near_duplicate_groups(units, {"ngram_range": [3, 5], "group_threshold": 0.7,
                                                 "review_threshold": 0.6})
    assert group["b-1"] == "a-1" and group["a-1"] == "a-1" and group["c-1"] == "c-1"
    assert set(pairs["id_a"]) | set(pairs["id_b"]) == {"a-1", "b-1"}
