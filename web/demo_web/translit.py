"""Translittération Arabizi -> alphabet arabe par règles (palier Bronze : déterministe, sans ressource externe).

Les règles ont été écrites a priori à partir des conventions de l'Arabizi tunisien (chiffres pour les sons
absents du français, digraphes, voyelles brèves non écrites), puis vérifiées une seule fois sur la moitié « dev »
de TUNIZI. La moitié « test » de TUNIZI ne sert qu'à la mesure finale.
"""
from __future__ import annotations

import re

# Ordre important : les digraphes passent avant les lettres seules.
DIGRAPHS = (("ch", "ش"), ("sh", "ش"), ("kh", "خ"), ("gh", "غ"), ("th", "ث"), ("dh", "ذ"),
            ("ou", "و"), ("oo", "و"), ("ee", "ي"), ("ii", "ي"))
DIGITS = {"2": "ء", "3": "ع", "5": "خ", "6": "ط", "7": "ح", "8": "غ", "9": "ق"}
CONSONANTS = {"b": "ب", "t": "ت", "j": "ج", "d": "د", "r": "ر", "z": "ز", "s": "س", "f": "ف",
              "q": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن", "h": "ه", "w": "و", "y": "ي",
              "g": "ق", "v": "ف", "p": "ب", "c": "ك", "x": "كس"}
VOWELS = frozenset("aeiou")
WORD = re.compile(r"[a-z0-9]+")
LATIN_LETTER = re.compile(r"[a-z]")
REPEATS = re.compile(r"(.)\1{2,}")


def translitterer_mot(mot: str) -> str:
    """Un mot Arabizi (minuscules) -> arabe. Un nombre isolé reste inchangé."""
    if not LATIN_LETTER.search(mot):
        return mot
    mot = REPEATS.sub(r"\1\1", mot)                      # lettres répétées 3 fois ou plus -> 2
    out, i, n = [], 0, len(mot)
    while i < n:
        pair = mot[i:i + 2]
        hit = next((arabe for digraphe, arabe in DIGRAPHS if pair == digraphe), None)
        if hit:
            out.append(hit)
            i += 2
            continue
        ch = mot[i]
        if ch in DIGITS:
            out.append(DIGITS[ch])
        elif ch in CONSONANTS:
            if not (out and out[-1] == CONSONANTS[ch] and ch != "h"):   # consonne doublée -> une seule
                out.append(CONSONANTS[ch])
        elif ch in VOWELS:
            if i == 0:
                out.append("ا")                            # voyelle initiale -> alef
            elif ch == "i":
                out.append("ي")                            # i -> ي
            elif ch == "a" and i == n - 1:
                out.append("ا")                            # a final -> ا
            # autres voyelles brèves : non écrites
        i += 1
    return "".join(out)


def translitterer(texte: str) -> str:
    """Translittère chaque mot latin d'un texte ; le reste (arabe, ponctuation, émojis) est conservé."""
    return WORD.sub(lambda m: translitterer_mot(m.group(0)), texte.lower())
