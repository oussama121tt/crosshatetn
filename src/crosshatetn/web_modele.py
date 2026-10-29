"""Inférence de la démo en ligne (copiée dans web/api/_lib/modele.py par export-web) avec numpy seul (sans scikit-learn, trop lourd pour une fonction Vercel).

Reproduit exactement TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
suivi de LinearSVC.decision_function. Le modèle (vocabulaire, idf, poids) est produit par
`python -m crosshatetn export-web`, qui vérifie l'égalité des scores avec scikit-learn avant d'écrire.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import text
from .translit import translitterer

HATE, NON_HATE = "haineux", "non_haineux"
MAX_CHARS = 500
TOP_K = 6
NEAR_ZERO = 0.1
FRENCH_LIMIT = 0.2
_WHITE_SPACES = re.compile(r"\s\s+")


class Modele:
    def __init__(self, chemin: Path):
        z = np.load(chemin, allow_pickle=False)
        self.vocab = {g: i for i, g in enumerate(z["vocab"].tolist())}
        self.names = z["vocab"]
        self.idf = z["idf"]
        self.coef = z["coef"]                 # déjà orienté : poids > 0 → haineux
        self.intercept = float(z["intercept"])
        self.ngram_min, self.ngram_max = (int(v) for v in z["ngram_range"])
        self.description = str(z["description"])

    def _ngrams(self, doc: str) -> list[str]:
        doc = _WHITE_SPACES.sub(" ", doc.lower())
        out = []
        for w in doc.split():
            w = " " + w + " "
            for n in range(self.ngram_min, self.ngram_max + 1):
                offset = 0
                out.append(w[offset:offset + n])
                while offset + n < len(w):
                    offset += 1
                    out.append(w[offset:offset + n])
                if offset == 0:
                    break
        return out

    def run(self, doc: str) -> dict:
        counts = Counter(g for g in self._ngrams(doc.strip()) if g in self.vocab)
        if counts:
            idx = np.fromiter((self.vocab[g] for g in counts), dtype=np.int64)
            tf = np.fromiter(counts.values(), dtype=np.float64)
            val = (1.0 + np.log(tf)) * self.idf[idx]
            val /= np.linalg.norm(val)
        else:
            idx, val = np.zeros(0, dtype=np.int64), np.zeros(0)
        contrib = self.coef[idx] * val
        score = float(contrib.sum() + self.intercept)
        order = np.argsort(contrib)
        vers_h = [(str(self.names[idx[i]]), float(contrib[i])) for i in order[::-1] if contrib[i] > 0][:TOP_K]
        vers_n = [(str(self.names[idx[i]]), float(contrib[i])) for i in order if contrib[i] < 0][:TOP_K]
        return {
            "score": round(score, 3),
            "decision": HATE if score > 0 else NON_HATE,
            "proche_frontiere": abs(score) < NEAR_ZERO,
            "ngrammes": {
                "nombre": int(len(idx)),
                "vers_haineux": [{"ngramme": g, "poids": round(w, 3)} for g, w in vers_h],
                "vers_non_haineux": [{"ngramme": g, "poids": round(w, 3)} for g, w in vers_n],
            },
        }


def check_input(raw: str) -> tuple[str, str]:
    stripped = raw.strip()
    if not stripped:
        return "vide", "Entrée vide : rien à classer."
    if len(stripped) > MAX_CHARS:
        return "trop_long", f"Plus de {MAX_CHARS} caractères : la démo refuse les textes longs."
    if not any(ch.isalpha() for ch in stripped):
        return "hors_perimetre", "Aucune lettre : émojis, chiffres ou ponctuation seuls."
    script = text.detect_script(stripped)
    if script == "other":
        return "hors_perimetre", "Alphabet non pris en charge : le modèle lit l'arabe et l'Arabizi latin."
    if script == "latin" and not text.has_arabizi_digits(stripped) and text.french_ratio(stripped) >= FRENCH_LIMIT:
        return "hors_perimetre", "Texte probablement en français : le modèle est entraîné sur le tunisien."
    return "ok", ""


def classify(raw: str, modele: Modele) -> dict:
    """Même trace que crosshatetn.demo.classify."""
    t0 = time.perf_counter()
    status, message = check_input(raw)
    trace: dict = {"entree": raw, "statut": status, "message": message}
    if status != "ok":
        trace["decision"] = "abstention"
    else:
        stripped = raw.strip()
        script = text.detect_script(stripped)
        lu = translitterer(stripped) if script in ("latin", "mixed") else stripped
        trace["ecriture"] = script
        trace["arabizi_chiffres"] = text.has_arabizi_digits(stripped)
        trace["normalisation"] = {"brut": stripped, "legere": text.normalize_light(lu),
                                  "forte": text.normalize_strong(lu),
                                  "utilise_par_le_modele": "brut" if lu == stripped else "translittéré"}
        trace.update(modele.run(lu))
        if lu != stripped:
            sans = modele.run(stripped)
            trace["translitteration"] = {"statut": "regles", "arabe": lu,
                                         "score_sans": sans["score"], "decision_sans": sans["decision"]}
    trace["latence_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    trace["modele"] = modele.description
    trace["avertissement"] = "Aide à la revue humaine : jamais une sanction automatique."
    return trace
