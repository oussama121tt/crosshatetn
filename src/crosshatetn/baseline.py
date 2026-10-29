"""E1 : baselines (majoritaire, TF-IDF + régression logistique, TF-IDF + SVM linéaire).

Règles du protocole :
- les choix (variante de texte, C, modèle) se font sur la validation tunisienne uniquement ;
- le test n'est évalué que par final_test(confirm=True), et chaque utilisation est journalisée ;
- les modèles linéaires sont déterministes : la variabilité est estimée par bootstrap.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from .config import Config
from .metrics import bootstrap_ci, confusion_table, paired_bootstrap, scores

TEST_JOURNAL = "journal_test.csv"


def build_features():
    """Mots (1-2) + caractères (2-5, robustes aux graphies instables), pondérés TF-IDF."""
    return make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    )


def build_classifier(kind: str, C: float | None, seed: int):
    if kind == "majoritaire":
        return DummyClassifier(strategy="most_frequent")
    if kind == "svm":
        return LinearSVC(C=C, class_weight="balanced", random_state=seed)
    if kind == "logreg":
        return LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
    raise ValueError(f"Modèle inconnu : {kind}")


def build_model(kind: str, C: float | None, seed: int):
    return make_pipeline(build_features(), build_classifier(kind, C, seed))


def select(labeled: pd.DataFrame, sources: list[str], split: str) -> pd.DataFrame:
    return labeled[labeled["source"].isin(sources) & (labeled["split"] == split)]


def _key(kind: str, train_name: str, variant: str, C) -> str:
    """Clé texte d'une configuration (évite les NaN comme clés de dictionnaire)."""
    c = "-" if C is None or (isinstance(C, float) and np.isnan(C)) else f"{float(C):g}"
    return f"{kind} | {train_name} | {variant} | C={c}"


@dataclass
class BaselineResults:
    grid: pd.DataFrame          # tous les essais, évalués sur la validation cible
    best: pd.DataFrame          # meilleur C par (modèle, entraînement, variante) + IC bootstrap
    comparisons: pd.DataFrame   # bootstraps appariés
    reference: dict             # configuration de référence E1 (SVM entraîné sur le tunisien)
    report: str                 # rapport de classification de la référence
    confusion: pd.DataFrame
    errors: pd.DataFrame


def run(labeled: pd.DataFrame, cfg: Config, out_dir: Path | None = None) -> BaselineResults:
    bcfg, labels, seed = cfg.section("baseline"), cfg.labels, cfg.seed
    target = bcfg["eval_source"]
    val = select(labeled, [target], "validation")
    y_val = val["label"].to_numpy()
    rows, preds = [], {}

    def record(kind, train_name, variant, C, pred, seconds):
        preds[_key(kind, train_name, variant, C)] = pred
        s = scores(y_val, pred, labels)
        rows.append({"modele": kind, "entrainement": train_name, "texte": variant, "C": C,
                     "macro_f1": s["macro_f1"], "f1_hate": s["f1_hate"], "rappel_hate": s["recall_hate"],
                     "secondes": round(seconds, 2)})

    tn_train = select(labeled, [target], "train")
    start = time.perf_counter()
    dummy = build_classifier("majoritaire", None, seed).fit(tn_train[["text_raw"]], tn_train["label"])
    record("majoritaire", "tn", "text_raw", None, dummy.predict(val[["text_raw"]]), time.perf_counter() - start)

    for train_name, sources in bcfg["train_sets"].items():
        train = select(labeled, sources, "train")
        for variant in bcfg["text_variants"]:
            features = build_features().fit(train[variant])          # ajusté sur l'entraînement seul
            X_train, X_val = features.transform(train[variant]), features.transform(val[variant])
            for kind, grid in bcfg["models"].items():
                for C in grid:
                    start = time.perf_counter()
                    clf = build_classifier(kind, C, seed).fit(X_train, train["label"])
                    record(kind, train_name, variant, float(C), clf.predict(X_val), time.perf_counter() - start)

    grid = pd.DataFrame(rows)
    order = grid.sort_values(["macro_f1", "C"], ascending=[False, True])
    best = order.drop_duplicates(["modele", "entrainement", "texte"]).copy()
    n_boot = int(bcfg["bootstrap"])
    cis = [bootstrap_ci(y_val, preds[_key(r.modele, r.entrainement, r.texte, r.C)], labels, n_boot, seed)
           for r in best.itertuples()]
    best["ic95_bas"], best["ic95_haut"] = [c[1] for c in cis], [c[2] for c in cis]
    best = best.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    def best_key(kind, train_name, variant=None):
        sub = best[(best["modele"] == kind) & (best["entrainement"] == train_name)]
        if variant is not None:
            sub = sub[sub["texte"] == variant]
        r = sub.iloc[0]
        return _key(r["modele"], r["entrainement"], r["texte"], r["C"]), r

    ref, ref_row = best_key("svm", "tn")
    comparisons = []

    def compare(name, key_a, key_b):
        res = paired_bootstrap(y_val, preds[key_a], preds[key_b], labels, n_boot, seed)
        comparisons.append({"comparaison": name, "A": key_a, "B": key_b, **res})

    if "tn+lev" in bcfg["train_sets"]:
        compare("E2 (aperçu) : + levantin vs tunisien seul", best_key("svm", "tn+lev")[0], ref)
    for variant in bcfg["text_variants"]:
        if variant != "text_raw":
            compare(f"E3 (aperçu) : {variant} vs text_raw", best_key("svm", "tn", variant)[0],
                    best_key("svm", "tn", "text_raw")[0])
    compare("régression logistique vs SVM", best_key("logreg", "tn")[0], ref)
    compare("SVM vs majoritaire", ref, _key("majoritaire", "tn", "text_raw", None))
    comparisons = pd.DataFrame(comparisons)

    ref_pred = preds[ref]
    reference = {"modele": ref_row["modele"], "entrainement": ref_row["entrainement"], "texte": ref_row["texte"],
                 "C": float(ref_row["C"]), "sources_entrainement": list(bcfg["train_sets"][ref_row["entrainement"]])}
    report = classification_report(y_val, ref_pred, labels=labels, digits=3, zero_division=0)
    confusion = confusion_table(y_val, ref_pred, labels)
    errors = (val.assign(prediction=ref_pred)
              .loc[lambda d: d["prediction"] != d["label"], ["id", "label", "prediction", "text_raw"]]
              .assign(categorie_erreur="", commentaire=""))

    results = BaselineResults(grid, best, comparisons, reference, report, confusion, errors)
    if out_dir is not None:
        save(results, Path(out_dir))
    return results


def save(results: BaselineResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.grid.to_csv(out_dir / "grille_validation.csv", index=False)
    results.best.to_csv(out_dir / "meilleurs_validation.csv", index=False)
    results.comparisons.to_csv(out_dir / "comparaisons_validation.csv", index=False)
    results.errors.to_csv(out_dir / "erreurs_validation_a_categoriser.csv", index=False)
    results.confusion.to_csv(out_dir / "confusion_validation.csv")
    (out_dir / "reference_e1.json").write_text(json.dumps(results.reference, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    (out_dir / "rapport_validation.txt").write_text(results.report, encoding="utf-8")


def final_test(labeled: pd.DataFrame, cfg: Config, reference: dict, confirm: bool = False) -> dict:
    """Évalue UNE configuration figée sur le test gelé. Chaque appel est journalisé dans splits/."""
    if not confirm:
        raise PermissionError("Évaluation sur le test refusée : passer confirm=True une fois tous les choix figés.")
    journal = cfg.path("splits") / TEST_JOURNAL
    previous = pd.read_csv(journal) if journal.exists() else pd.DataFrame()
    if len(previous):
        print(f"Attention : le test a déjà été utilisé {len(previous)} fois (voir {journal}).")
    labels, seed, target = cfg.labels, cfg.seed, cfg.section("baseline")["eval_source"]
    train = select(labeled, reference["sources_entrainement"], "train")
    test = select(labeled, [target], "test")
    variant = reference["texte"]
    model = build_model(reference["modele"], reference["C"], seed).fit(train[variant], train["label"])
    pred = model.predict(test[variant])
    point, low, high = bootstrap_ci(test["label"], pred, labels, int(cfg.section("baseline")["bootstrap"]), seed)
    result = {**reference, **scores(test["label"], pred, labels), "ic95_bas": low, "ic95_haut": high,
              "n_test": int(len(test))}
    entry = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
             "configuration": json.dumps(reference, ensure_ascii=False), "macro_f1": round(point, 4)}
    write_header = not journal.exists()
    with open(journal, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(entry))
        if write_header:
            writer.writeheader()
        writer.writerow(entry)
    return result
