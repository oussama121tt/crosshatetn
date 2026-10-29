"""Contrôles de fuite entre splits.

Un contrôle bloquant qui échoue lève LeakageError : on ne doit jamais entraîner sur des splits qui fuient.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .splits import SPLITS


class LeakageError(AssertionError):
    """Fuite détectée entre splits."""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    blocking: bool = True


@dataclass
class AuditResult:
    checks: list[Check]
    review_pairs: pd.DataFrame

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.blocking)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "contrôle": c.name,
            "résultat": "OK" if c.ok else ("ÉCHEC" if c.blocking else "à revoir"),
            "bloquant": "oui" if c.blocking else "non",
            "détail": c.detail,
        } for c in self.checks])


def _shared_across_splits(df: pd.DataFrame, column: str) -> int:
    return int((df.groupby(column)["split"].nunique() > 1).sum())


def run_audit(labeled: pd.DataFrame, pairs: pd.DataFrame, group_threshold: float, review_threshold: float,
              max_class_gap: float) -> AuditResult:
    """labeled : textes étiquetés uniques avec id, source, label, split, group, text_raw, dedup_key.
    pairs : paires (id_a, id_b, sim) de similarité >= seuil de revue, toutes sources confondues."""
    checks = []
    n_dup = int(labeled["id"].duplicated().sum())
    checks.append(Check("identifiants uniques", n_dup == 0, f"{n_dup} doublon(s) d'identifiant"))

    bad_split = int((~labeled["split"].isin(SPLITS)).sum())
    checks.append(Check("un split valide par texte", bad_split == 0, f"{bad_split} texte(s) sans split valide"))

    n = _shared_across_splits(labeled, "group")
    checks.append(Check("groupes de quasi-doublons dans un seul split", n == 0, f"{n} groupe(s) partagé(s)"))

    n = _shared_across_splits(labeled, "text_raw")
    checks.append(Check("aucun texte brut identique entre splits", n == 0, f"{n} texte(s) partagé(s)"))

    n = _shared_across_splits(labeled, "dedup_key")
    checks.append(Check("aucune clé normalisée identique entre splits", n == 0, f"{n} clé(s) partagée(s)"))

    split_of = labeled.set_index("id")["split"]
    p = pairs[pairs["id_a"].isin(split_of.index) & pairs["id_b"].isin(split_of.index)].copy()
    p["split_a"], p["split_b"] = p["id_a"].map(split_of), p["id_b"].map(split_of)
    cross = p[p["split_a"] != p["split_b"]]
    n_strong = int((cross["sim"] >= group_threshold).sum())
    checks.append(Check(f"aucune paire de similarité ≥ {group_threshold} entre splits", n_strong == 0,
                        f"{n_strong} paire(s)"))
    review = cross[(cross["sim"] >= review_threshold) & (cross["sim"] < group_threshold)]
    checks.append(Check(f"paires dans [{review_threshold} ; {group_threshold}[ entre splits", len(review) == 0,
                        f"{len(review)} paire(s) listée(s) pour revue humaine", blocking=False))

    worst = []
    for source, sub in labeled.groupby("source"):
        overall = sub["label"].value_counts(normalize=True)
        for split, part in sub.groupby("split"):
            gap = (part["label"].value_counts(normalize=True).reindex(overall.index, fill_value=0) - overall).abs()
            worst.append((gap.max(), f"{source}/{split}/{gap.idxmax()}"))
    gap, where = max(worst) if worst else (0.0, "-")
    checks.append(Check(f"proportions de classes par split (écart ≤ {max_class_gap:.0%})", gap <= max_class_gap,
                        f"écart maximal {gap:.1%} ({where})", blocking=False))

    sizes = labeled["group"].value_counts()
    checks.append(Check("taille des groupes de quasi-doublons", True,
                        f"{int((sizes > 1).sum())} groupe(s) de 2+ textes, le plus grand en compte {int(sizes.max())}",
                        blocking=False))

    texts = labeled.set_index("id")["text_raw"]
    review_pairs = review.assign(texte_a=review["id_a"].map(texts), texte_b=review["id_b"].map(texts))
    return AuditResult(checks=checks, review_pairs=review_pairs.sort_values("sim", ascending=False))
