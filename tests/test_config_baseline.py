import copy
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from crosshatetn import baseline
from crosshatetn.config import Config, ConfigError, load_config, validate

ROOT = Path(__file__).resolve().parents[1]


def test_repository_config_is_valid():
    cfg = load_config(ROOT / "configs" / "data.yaml")
    assert cfg.root == ROOT
    assert set(cfg.sources) == {"thsab", "lhsab", "tunizi"}


def test_invalid_thresholds_are_rejected():
    cfg = load_config(ROOT / "configs" / "data.yaml")
    data = copy.deepcopy(cfg.data)
    data["near_duplicates"]["review_threshold"] = 0.9  # au-dessus du seuil de groupe : incohérent
    try:
        validate(Config(data=data, root=cfg.root))
    except ConfigError:
        return
    raise AssertionError("des seuils incohérents doivent être refusés")


def toy_labeled(n=240, seed=0):
    rng = np.random.default_rng(seed)
    vocab = {"normal": ["salam", "behi", "يعطيك الصحة", "مرحبا"], "abusive": ["كلب", "حمار", "mahboul"],
             "hate": ["اقتلوهم", "يهود", "kahlouch"]}
    rows = []
    for i in range(n):
        label = rng.choice(["normal", "normal", "abusive", "hate"])
        text = " ".join(rng.choice(vocab[label], 2)) + f" {rng.integers(100)}"
        source = "thsab" if i % 3 else "lhsab"
        split = ["train", "train", "train", "validation", "test"][i % 5]
        rows.append({"id": f"{source}-{i:05d}", "source": source, "label": label, "split": split,
                     "text_raw": text, "text_light": text, "text_strong": text.lower()})
    return pd.DataFrame(rows)


def toy_config(tmp: str) -> Config:
    cfg = load_config(ROOT / "configs" / "data.yaml")
    data = copy.deepcopy(cfg.data)
    data["baseline"].update({"models": {"svm": [1.0], "logreg": [1.0]}, "bootstrap": 50})
    data["paths"]["splits"] = tmp
    return Config(data=data, root=cfg.root)


def test_baseline_runs_end_to_end_on_toy_data():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = toy_config(tmp)
        results = baseline.run(toy_labeled(), cfg, Path(tmp) / "out")
        assert {"svm", "logreg", "majoritaire"} <= set(results.best["modele"])
        assert (results.best["ic95_bas"] <= results.best["macro_f1"]).all()
        assert results.reference["modele"] == "svm" and results.reference["entrainement"] == "tn"
        assert (Path(tmp) / "out" / "reference_e1.json").exists()


def test_final_test_requires_confirmation_and_is_logged():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, data = toy_config(tmp), toy_labeled()
        reference = {"modele": "svm", "entrainement": "tn", "texte": "text_raw", "C": 1.0,
                     "sources_entrainement": ["thsab"]}
        try:
            baseline.final_test(data, cfg, reference)
        except PermissionError:
            pass
        else:
            raise AssertionError("le test ne doit pas être évalué sans confirmation")
        assert not (Path(tmp) / baseline.TEST_JOURNAL).exists()
        result = baseline.final_test(data, cfg, reference, confirm=True)
        assert 0 <= result["macro_f1"] <= 1
        journal = pd.read_csv(Path(tmp) / baseline.TEST_JOURNAL)
        assert len(journal) == 1
