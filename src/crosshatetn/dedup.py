"""Doublons, conflits d'étiquettes et groupes de quasi-doublons.

Statuts possibles d'une ligne :
  ok       conservée
  vide     texte vide (aucun contenu exploitable)
  doublon  même clé normalisée qu'une ligne conservée de la même source
  conflit  représentant d'un groupe de doublons aux étiquettes contradictoires, non arbitré
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ARBITRATION_COLUMNS = ["id_representant", "label_arbitre", "arbitre", "date", "commentaire"]


def read_arbitration(path: Path, labels: list[str]) -> pd.DataFrame:
    """Lit les décisions humaines (fichier versionné). Une étiquette inconnue est une erreur."""
    if not Path(path).exists():
        return pd.DataFrame(columns=ARBITRATION_COLUMNS)
    arb = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ARBITRATION_COLUMNS if c not in arb.columns]
    if missing:
        raise ValueError(f"{path} : colonnes manquantes {missing}")
    arb = arb[arb["label_arbitre"].str.strip() != ""].copy()
    arb["label_arbitre"] = arb["label_arbitre"].str.strip()
    wrong = sorted(set(arb["label_arbitre"]) - set(labels))
    if wrong:
        raise ValueError(f"{path} : étiquettes inconnues {wrong} (attendu : {labels})")
    if arb["id_representant"].duplicated().any():
        raise ValueError(f"{path} : un même représentant est arbitré deux fois")
    return arb


def mark_duplicates(corpus: pd.DataFrame) -> pd.DataFrame:
    """Regroupe, par source, les lignes de même clé normalisée.

    Le représentant d'un groupe est la ligne de plus petit rang dans le fichier source.
    """
    out = corpus.copy()
    out["status"] = np.where(out["dedup_key"] == "", "vide", "ok")
    usable = out["status"] == "ok"
    order = out[usable].sort_values(["source", "source_row"])
    representative = order.groupby(["source", "dedup_key"])["id"].transform("first")
    out["dup_group"] = None
    out.loc[representative.index, "dup_group"] = representative
    out["n_in_group"] = out.groupby("dup_group")["id"].transform("size").where(usable)
    duplicate = usable & (out["dup_group"] != out["id"])
    out.loc[duplicate, "status"] = "doublon"
    return out


def resolve_conflicts(corpus: pd.DataFrame, arbitration: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repère les groupes aux étiquettes contradictoires et applique arbitrage ou politique.

    Renvoie le corpus mis à jour et la table des conflits (pour l'arbitrage humain).
    """
    out = corpus.copy()
    out["label_status"] = np.where(out["label"].notna(), "source", "")
    labeled = out[(out["role"] == "labeled") & out["dup_group"].notna()]
    decisions = dict(zip(arbitration["id_representant"], arbitration["label_arbitre"]))
    rows = []
    for rep_id, group in labeled.groupby("dup_group", sort=True):
        counts = Counter(group["label"])
        if len(counts) < 2:
            continue
        (top, top_n), *rest = counts.most_common()
        majority = top if top_n > rest[0][1] else ""
        if rep_id in decisions:
            resolution, label = "arbitre", decisions[rep_id]
        elif policy == "majority" and majority:
            resolution, label = "majorite", majority
        else:
            resolution, label = "exclu", None
        rep_mask = out["id"] == rep_id
        if label is None:
            out.loc[rep_mask, "status"] = "conflit"
        else:
            out.loc[rep_mask, ["label", "label_status"]] = [label, resolution]
        rows.append({
            "id_representant": rep_id,
            "source": group["source"].iat[0],
            "membres": " ".join(group.sort_values("source_row")["id"]),
            "etiquettes": ", ".join(f"{lab}×{n}" for lab, n in sorted(counts.items())),
            "majorite_stricte": majority,
            "resolution": resolution,
            "label_retenu": label or "",
            "texte": group.loc[group["id"] == rep_id, "text_light"].iat[0],
        })
    conflicts = pd.DataFrame(rows, columns=["id_representant", "source", "membres", "etiquettes", "majorite_stricte",
                                            "resolution", "label_retenu", "texte"])
    unknown = sorted(set(decisions) - set(conflicts["id_representant"]))
    if unknown:
        print(f"Attention : {len(unknown)} arbitrage(s) ne correspondent à aucun conflit actuel : {unknown[:5]}")
    return out, conflicts


def similar_pairs(texts: list[str], ngram_range: tuple[int, int], min_sim: float, block: int = 1000) -> pd.DataFrame:
    """Toutes les paires (i < j) dont la similarité cosinus des n-grammes de caractères est >= min_sim."""
    if len(texts) < 2:
        return pd.DataFrame({"i": [], "j": [], "sim": []})
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(ngram_range), sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(texts)
    found = []
    for start in range(0, X.shape[0], block):
        sims = (X[start:start + block] @ X.T).toarray()
        rows, cols = np.nonzero(sims >= min_sim)
        keep = (start + rows) < cols
        found.append(np.column_stack([start + rows[keep], cols[keep], sims[rows[keep], cols[keep]]]))
    pairs = np.concatenate(found) if found else np.empty((0, 3))
    return pd.DataFrame({"i": pairs[:, 0].astype(int), "j": pairs[:, 1].astype(int),
                         "sim": np.minimum(pairs[:, 2], 1.0).round(4)})


def connected_groups(n: int, pairs: pd.DataFrame) -> np.ndarray:
    """Composantes connexes (union-find) : renvoie pour chaque élément l'indice du plus petit élément de son groupe."""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(pairs["i"], pairs["j"]):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def near_duplicate_groups(units: pd.DataFrame, cfg_nd: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Groupes de quasi-doublons sur les textes uniques (toutes sources confondues).

    Renvoie l'identifiant de groupe de chaque ligne (id du plus petit membre, ordre stable)
    et les paires dont la similarité dépasse le seuil de revue.
    """
    units = units.sort_values("id").reset_index(drop=True)
    pairs = similar_pairs(units["text_light"].tolist(), cfg_nd["ngram_range"], cfg_nd["review_threshold"])
    strong = pairs[pairs["sim"] >= cfg_nd["group_threshold"]]
    roots = connected_groups(len(units), strong)
    group = pd.Series(units["id"].to_numpy()[roots], index=units["id"], name="group")
    named = pd.DataFrame({
        "id_a": units["id"].to_numpy()[pairs["i"]],
        "id_b": units["id"].to_numpy()[pairs["j"]],
        "sim": pairs["sim"].to_numpy(),
    })
    return group, named
