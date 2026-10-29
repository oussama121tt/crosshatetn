from crosshatetn.text import (DIACRITICS, dedup_key, detect_script, french_ratio, has_arabizi_digits,
                              normalize_light, normalize_strong)

FATHA, SUKUN, FATHATAN, SHADDA = chr(0x064E), chr(0x0652), chr(0x064B), chr(0x0651)
TATWEEL = chr(0x0640)
SAMPLES = ["مرحبا بيكم", "sa7a chbik 3lik", "تونس 😂!!", "  ﻻ   بأس ", "أحمد إلى آمال", "فساااااد", ""]


def test_light_removes_diacritics_and_tatweel():
    assert normalize_light("م" + FATHA + "ر" + SUKUN + "ح" + FATHA + "ب" + FATHATAN + "ا") == "مرحبا"
    assert normalize_light("جمي" + TATWEEL * 3 + "ل") == "جميل"
    assert normalize_light("مد" + SHADDA + "ة") == "مدة"


def test_light_unifies_alef_variants():
    assert normalize_light("أحمد إلى آمال ٱلله") == "احمد الى امال الله"


def test_light_applies_nfkc_and_whitespace():
    # « ﻻ » est une forme de présentation (ligature lam-alef) : NFKC la décompose en « لا »
    assert normalize_light("  ﻻ   بأس \n") == "لا باس"


def test_light_keeps_emoji_punctuation_and_case():
    assert normalize_light("تونس 😂!! OK") == "تونس 😂!! OK"


def test_normalizations_are_idempotent():
    for fn in (normalize_light, normalize_strong, dedup_key):
        for s in SAMPLES:
            assert fn(fn(s)) == fn(s), (fn.__name__, s)


def test_strong_normalization():
    assert normalize_strong("مدرسة على") == "مدرسه علي"
    assert normalize_strong("فساااااد") == "فسااد"
    assert normalize_strong("SALUT hhhhh") == "salut hh"


def test_dedup_key_merges_surface_variants():
    assert dedup_key("التوانسة فساااااد !!!") == dedup_key("التوانسة فسااد")
    assert dedup_key("هههههه") == dedup_key("ههه")
    assert dedup_key("تونس 😂") == dedup_key("تونس")


def test_dedup_key_keeps_word_order():
    # Un changement d'ordre n'est pas un doublon exact : il est traité par les quasi-doublons
    assert dedup_key("قليل تربية لطفي") != dedup_key("لطفي قليل تربية")


def test_diacritics_class_is_exact():
    for code in range(0x064B, 0x0660):
        assert DIACRITICS.fullmatch(chr(code)), hex(code)
    for code in range(0x0621, 0x064B):  # lettres arabes : jamais supprimées
        assert not DIACRITICS.fullmatch(chr(code)), hex(code)


def test_detect_script():
    assert detect_script("مرحبا بيكم") == "arabic"
    assert detect_script("sa7a chbik") == "latin"
    assert detect_script("مرحبا ok") == "mixed"
    assert detect_script("😂 123 !!") == "other"
    assert detect_script("مرحبا بيكم ï") == "arabic"  # une lettre parasite isolée ne fait pas un texte mixte


def test_arabizi_digits():
    assert has_arabizi_digits("sa7a chbik")
    assert has_arabizi_digits("3ijbitni")
    assert not has_arabizi_digits("2020 winkom")
    assert not has_arabizi_digits("مرحبا 3")


def test_french_ratio():
    assert french_ratio("merci pour tout le monde") >= 0.5
    assert french_ratio("rabi yarhmou") == 0.0
    assert french_ratio("") == 0.0
