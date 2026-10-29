import pandas as pd

from crosshatetn.demo import MAX_CHARS, TOP_K, classify, fit_model


def toy_model():
    hate = ["يا بهيم سكر فمك", "انت بهيم يا حمار", "اسكت يا بهيم", "يا حمار جاهل", "بهيم وحمار"] * 4
    ok = ["برافو عليك ربي يوفقك", "ربي يهديكم جميعا", "مبروك النجاح يا صاحبي", "الله يعطيك الصحة"] * 4
    train = pd.DataFrame({"text_raw": hate + ok, "label": ["abusive"] * len(hate) + ["normal"] * len(ok)})
    return fit_model(train, seed=42, description="test")


def test_entree_vide_donne_une_abstention():
    r = classify("   ", toy_model())
    assert r["statut"] == "vide" and r["decision"] == "abstention" and "score" not in r


def test_texte_trop_long_est_refuse():
    r = classify("ا" * (MAX_CHARS + 1), toy_model())
    assert r["statut"] == "trop_long" and r["decision"] == "abstention"


def test_sans_lettre_est_hors_perimetre():
    assert classify("123 !!! 😂", toy_model())["statut"] == "hors_perimetre"


def test_alphabet_non_pris_en_charge():
    assert classify("你好世界", toy_model())["statut"] == "hors_perimetre"


def test_francais_est_hors_perimetre():
    r = classify("Bonjour, je vais bien merci à tous les amis", toy_model())
    assert r["statut"] == "hors_perimetre"


def test_arabe_donne_score_et_classe_coherents():
    r = classify("يا بهيم سكر فمك", toy_model())
    assert r["statut"] == "ok"
    assert (r["score"] > 0) == (r["decision"] == "haineux")
    assert r["ecriture"] == "arabic"


def test_trace_contient_les_etapes_et_les_ngrammes():
    r = classify("يا بهيم سكر فمك", toy_model())
    assert set(r["normalisation"]) == {"brut", "legere", "forte", "utilise_par_le_modele"}
    assert r["normalisation"]["utilise_par_le_modele"] == "brut"
    assert r["ngrammes"]["nombre"] > 0
    assert 0 < len(r["ngrammes"]["vers_haineux"]) <= TOP_K
    assert all(item["poids"] > 0 for item in r["ngrammes"]["vers_haineux"])
    assert all(item["poids"] < 0 for item in r["ngrammes"]["vers_non_haineux"])
    assert r["latence_ms"] >= 0


def test_arabizi_est_translittere_par_regles():
    r = classify("ya bhim sakker fommek", toy_model())
    assert r["statut"] == "ok"
    assert r["translitteration"]["statut"] == "regles"
    assert r["translitteration"]["arabe"] == "يا بهيم سكر فمك"
    assert r["normalisation"]["utilise_par_le_modele"] == "translittéré"
    # la décision porte sur le texte translittéré
    assert r["score"] == classify("يا بهيم سكر فمك", toy_model())["score"]


def test_arabe_sans_translitteration():
    assert "translitteration" not in classify("يا بهيم سكر فمك", toy_model())
