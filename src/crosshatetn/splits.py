"""Découpage stratifié par groupes et gel de l'affectation.

Le fichier splits/affectation.csv (id, split) est la référence versionnée :
- une affectation existante n'est jamais modifiée (sauf re-gel explicite) ;
- un nouvel identifiant hérite du split de son groupe de quasi-doublons, sinon il est placé
  de façon déterministe (hachage) selon les proportions visées ;
- un groupe réparti sur plusieurs splits est une erreur.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

SPLITS = ("train", "validation", "test")
ASSIGNMENT_FILE = "affectation.csv"
MANIFEST_FILE = "manifest.json"


class SplitError(RuntimeError):
    """Affectation incohérente (fuite potentielle)."""


@dataclass
class SplitOutcome:
    assignment: pd.Series                     # index : id, valeurs : split
    created: bool                             # True si l'affectation vient d'être (re)créée
    added: pd.DataFrame = field(default_factory=pd.DataFrame)  # nouveaux ids ajoutés au fichier gelé
    stale_ids: list = field(default_factory=list)              # ids gelés absents des données actuelles


def _fold_counts(cfg_split: dict) -> tuple[int, int]:
    n_folds = int(cfg_split["n_folds"])
    n_test = round(cfg_split["test"] * n_folds)
    n_val = round(cfg_split["validation"] * n_folds)
    if n_test < 1 or n_val < 1 or n_test + n_val >= n_folds:
        raise SplitError("n_folds trop petit pour les proportions demandées")
    return n_test, n_val


def fresh_assignment(units: pd.DataFrame, cfg_split: dict, seed: int) -> pd.Series:
    """Affectation stratifiée (sur source × étiquette) et groupée (quasi-doublons ensemble)."""
    n_test, n_val = _fold_counts(cfg_split)
    strata = units[cfg_split["stratify_on"]].astype(str).agg("|".join, axis=1)
    sgkf = StratifiedGroupKFold(n_splits=int(cfg_split["n_folds"]), shuffle=True, random_state=seed)
    fold = np.full(len(units), -1)
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(units)), strata, groups=units["group"])):
        fold[idx] = k
    split = np.where(fold < n_test, "test", np.where(fold < n_test + n_val, "validation", "train"))
    return pd.Series(split, index=units["id"].to_numpy(), name="split")


def hashed_split(key: str, cfg_split: dict, seed: int) -> str:
    """Placement déterministe d'un nouveau groupe, indépendant de l'ordre d'arrivée."""
    value = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / 16 ** 12
    if value < cfg_split["test"]:
        return "test"
    if value < cfg_split["test"] + cfg_split["validation"]:
        return "validation"
    return "train"


def assign(units: pd.DataFrame, frozen: pd.Series | None, cfg_split: dict, seed: int,
           refreeze: bool = False) -> SplitOutcome:
    """Affecte un split à chaque texte étiqueté unique (colonnes requises : id, group + strates)."""
    if frozen is None or refreeze:
        return SplitOutcome(assignment=fresh_assignment(units, cfg_split, seed), created=True)

    current = units.set_index("id")
    split = current.index.to_series().map(frozen)
    by_group = (pd.DataFrame({"group": current["group"], "split": split})
                .dropna(subset=["split"]).groupby("group")["split"].agg(lambda s: sorted(set(s))))
    broken = by_group[by_group.map(len) > 1]
    if len(broken):
        raise SplitError(f"{len(broken)} groupe(s) de quasi-doublons répartis sur plusieurs splits "
                         f"(ex. {broken.index[0]} : {broken.iloc[0]}). Le regroupement a changé : "
                         "re-geler explicitement (refreeze) ou corriger le fichier gelé.")
    added = []
    for idx in split.index[split.isna()]:
        group = current.at[idx, "group"]
        inherited = by_group.get(group)
        new = inherited[0] if inherited else hashed_split(group, cfg_split, seed)
        split.at[idx] = new
        added.append({"id": idx, "split": new, "origine": "groupe" if inherited else "hachage"})
    stale = sorted(set(frozen.index) - set(current.index))
    return SplitOutcome(assignment=split.rename("split"), created=False,
                        added=pd.DataFrame(added, columns=["id", "split", "origine"]), stale_ids=stale)


def ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def load_frozen(directory: Path) -> pd.Series | None:
    path = Path(directory) / ASSIGNMENT_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str)
    if list(df.columns) != ["id", "split"] or not df["split"].isin(SPLITS).all() or df["id"].duplicated().any():
        raise SplitError(f"{path} : fichier d'affectation invalide")
    return df.set_index("id")["split"]


def load_manifest(directory: Path) -> dict | None:
    path = Path(directory) / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check_frozen_digests(frozen: pd.Series, manifest: dict | None) -> None:
    """Vérifie que le fichier gelé correspond à son manifeste (le test n'a pas été modifié à la main)."""
    if manifest is None:
        return
    for name in SPLITS:
        expected = manifest["digests"][name]
        found = ids_digest(frozen.index[frozen == name])
        if found != expected:
            raise SplitError(f"Le split '{name}' du fichier gelé ne correspond plus à son manifeste. "
                             "Il a été modifié hors du pipeline : restaurer la version versionnée.")


def save_frozen(directory: Path, outcome: SplitOutcome, frozen: pd.Series | None, units: pd.DataFrame,
                extra: dict) -> pd.Series:
    """Écrit l'affectation (création, re-gel ou ajouts) et met à jour le manifeste avec un historique."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if outcome.created or frozen is None:
        final = outcome.assignment
    else:
        final = pd.concat([frozen, outcome.assignment.loc[outcome.added["id"]]]) if len(outcome.added) else frozen
    final = final.sort_index()
    pd.DataFrame({"id": final.index, "split": final.to_numpy()}).to_csv(directory / ASSIGNMENT_FILE, index=False)

    previous = load_manifest(directory)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    history = (previous or {}).get("historique", [])
    if outcome.created:
        history = history + [{"date": now, "action": "re-gel" if previous else "création", "ids": int(len(final))}]
    elif len(outcome.added):
        history = history + [{"date": now, "action": "ajout", "ids": int(len(outcome.added)),
                              "par_split": outcome.added["split"].value_counts().to_dict()}]
    labels = units.set_index("id").reindex(final.index)
    counts = (pd.DataFrame({"split": final, "source": labels["source"], "label": labels["label"]})
              .dropna().groupby(["source", "split", "label"]).size())
    # Version de scikit-learn ayant produit le découpage (conservée tant qu'on ne re-gèle pas)
    created_with = (sklearn.__version__ if outcome.created or previous is None
                    else previous.get("scikit_learn_creation", sklearn.__version__))
    manifest = {
        "fichier": ASSIGNMENT_FILE,
        "effectifs": {name: int((final == name).sum()) for name in SPLITS},
        "detail": {f"{s}|{sp}|{lab}": int(n) for (s, sp, lab), n in counts.items()},
        "digests": {name: ids_digest(final.index[final == name]) for name in SPLITS},
        "scikit_learn_creation": created_with,
        **extra,
        "historique": history,
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return final
