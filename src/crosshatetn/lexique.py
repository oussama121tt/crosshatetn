"""Règle lexicale (palier Bronze) : un message est signalé haineux s'il contient un terme du lexique validé.

Le lexique part de candidats calculés sur l'entraînement T-HSAB uniquement (mots plus fréquents dans les
messages haineux), puis l'équipe garde à la main les vrais termes insultants ou haineux.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

from .text import normalize_light

TOKEN = re.compile(r"[^\W\d_]+")
PREFIXES = ("وال", "بال", "لل", "ال", "و", "ب", "ل", "ف", "يا")   # préfixes collés retirés avant la recherche


def tokens(texte: str) -> list[str]:
    return TOKEN.findall(normalize_light(texte))


def termes_trouves(texte: str, lexique: set[str]) -> list[str]:
    """Termes du lexique présents dans le texte (mot entier, ou mot précédé d'un préfixe collé)."""
    found = []
    for tok in tokens(texte):
        if tok in lexique:
            found.append(tok)
            continue
        for prefixe in PREFIXES:
            if tok.startswith(prefixe) and tok[len(prefixe):] in lexique:
                found.append(tok[len(prefixe):])
                break
    return found


def predire(texte: str, lexique: set[str]) -> str:
    return "haineux" if termes_trouves(texte, lexique) else "non_haineux"


def charger(termes) -> set[str]:
    """Normalise les termes validés comme les textes (normalisation légère), sans les vides."""
    return {normalize_light(str(t)).strip() for t in termes if str(t).strip()}


def candidats(textes, etiquettes, min_haineux: int = 5, k: int = 200) -> pd.DataFrame:
    """Mots plus fréquents dans les messages haineux (log-odds lissé sur les fréquences documentaires)."""
    dfh, dfn = Counter(), Counter()
    for texte, y in zip(textes, etiquettes):
        (dfh if y == "haineux" else dfn).update(set(tokens(texte)))
    n_h = sum(1 for y in etiquettes if y == "haineux")
    n_n = len(etiquettes) - n_h
    rows = []
    for mot in set(dfh) | set(dfn):
        h, n = dfh[mot], dfn[mot]
        if h < min_haineux or len(mot) < 2:
            continue
        rows.append((mot, h, n, math.log((h + 1) / (n_h + 2)) - math.log((n + 1) / (n_n + 2))))
    table = pd.DataFrame(rows, columns=["terme", "n_haineux", "n_non_haineux", "log_odds"])
    return table.sort_values(["log_odds", "terme"], ascending=[False, True]).head(k).reset_index(drop=True)
