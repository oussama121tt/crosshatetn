"""Normalisation réversible, clé de dédoublonnage et détection du système d'écriture.

Le texte brut n'est jamais modifié : chaque normalisation produit une colonne à part,
pour que la normalisation reste une variable d'expérience (E3).
Les caractères sont désignés par leur code Unicode (chr(0x...)) pour rester lisibles.
"""
from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas n'est importé que pour add_text_features : text.py reste utilisable sans pandas (démo web)
    import pandas as pd


def _char_class(*ranges: tuple[int, int]) -> str:
    parts = []
    for start, end in ranges:
        parts.append(re.escape(chr(start)) if start == end else f"{re.escape(chr(start))}-{re.escape(chr(end))}")
    return "[" + "".join(parts) + "]"


# Diacritiques arabes : signes coraniques, harakat (fatha, damma, kasra, tanwin, shadda, soukoun…),
# alef suscrit et marques d'annotation coranique.
DIACRITICS = re.compile(_char_class((0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670),
                                    (0x06D6, 0x06DC), (0x06DF, 0x06E8), (0x06EA, 0x06ED)))
TATWEEL = chr(0x0640)
ALEF = chr(0x0627)
ALEF_VARIANTS = re.compile(_char_class((0x0622, 0x0623), (0x0625, 0x0625), (0x0671, 0x0671)))  # آ أ إ ٱ
ALEF_MAKSURA, YEH = chr(0x0649), chr(0x064A)  # ى -> ي
TEH_MARBUTA, HEH = chr(0x0629), chr(0x0647)   # ة -> ه
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
WHITESPACE = re.compile(r"\s+")
REPEATS_3PLUS = re.compile(r"(.)\1{2,}")
REPEATS_2PLUS = re.compile(r"(.)\1+")
NON_WORD = re.compile(r"[^\w\s]|_")
# Chiffres utilisés comme lettres en Arabizi (2 ء, 3 ع, 5 خ, 6 ط, 7 ح, 8 غ, 9 ق), collés à une lettre latine
ARABIZI_DIGITS = re.compile(r"(?<=[A-Za-z])[2356789]|[2356789](?=[A-Za-z])")
FRENCH_STOPWORDS = frozenset(
    "le les de des du un une et est pour pas que qui dans je tu il elle nous vous ils sont avec sur mais "
    "tout tous au aux mon ton son mes tes ses cette ce ces merci bonne bon tres très bien comme plus être "
    "avoir fait aussi encore votre notre leur leurs ou où".split()
)


def normalize_light(text: str) -> str:
    """NFKC, sans diacritiques ni tatweel, variantes d'alef unifiées, espaces réduits."""
    t = unicodedata.normalize("NFKC", text)
    t = DIACRITICS.sub("", t).replace(TATWEEL, "")
    t = ALEF_VARIANTS.sub(ALEF, t)
    return WHITESPACE.sub(" ", t).strip()


def normalize_strong(text: str) -> str:
    """Normalisation légère + minuscules, ى→ي, ة→ه, lettres répétées 3 fois ou plus réduites à 2."""
    t = normalize_light(text).lower()
    t = t.replace(ALEF_MAKSURA, YEH).replace(TEH_MARBUTA, HEH)
    return REPEATS_3PLUS.sub(r"\1\1", t)


def dedup_key(text: str) -> str:
    """Clé agressive pour repérer les doublons : sans ponctuation, émojis ni lettres répétées.

    Elle sert uniquement à détecter les doublons, jamais d'entrée pour les modèles.
    """
    t = NON_WORD.sub(" ", normalize_strong(text))
    t = REPEATS_2PLUS.sub(r"\1", t)
    return WHITESPACE.sub(" ", t).strip()


def _is_arabic(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in ARABIC_RANGES)


def detect_script(text: str) -> str:
    """'arabic', 'latin', 'mixed' (au moins 2 lettres de chaque alphabet) ou 'other'.

    Mesure l'alphabet, pas le dialecte : un texte latin peut être de l'Arabizi ou du français.
    """
    arabic = latin = 0
    for ch in text:
        if ch.isalpha():
            if _is_arabic(ch):
                arabic += 1
            elif ch.isascii() or 0x00C0 <= ord(ch) <= 0x024F:
                latin += 1
    if arabic == 0 and latin == 0:
        return "other"
    if min(arabic, latin) >= 2:
        return "mixed"
    return "arabic" if arabic >= latin else "latin"


def has_arabizi_digits(text: str) -> bool:
    return ARABIZI_DIGITS.search(text) is not None


def french_ratio(text: str) -> float:
    """Part des mots outils français : simple indice, pas une identification de langue."""
    tokens = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", text.lower())
    return round(sum(tok in FRENCH_STOPWORDS for tok in tokens) / len(tokens), 3) if tokens else 0.0


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variantes normalisées, la clé de doublon et les indices d'écriture."""
    out = df.copy()
    raw = out["text_raw"]
    out["text_light"] = raw.map(normalize_light)
    out["text_strong"] = raw.map(normalize_strong)
    out["dedup_key"] = raw.map(dedup_key)
    out["script"] = raw.map(detect_script)
    out["arabizi_digits"] = raw.map(has_arabizi_digits)
    out["french_ratio"] = raw.map(french_ratio)
    out["n_chars"] = raw.str.len()
    return out
