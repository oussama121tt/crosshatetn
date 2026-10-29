from crosshatetn.lexique import candidats, charger, predire, termes_trouves
from crosshatetn.translit import translitterer, translitterer_mot


def test_chiffres_et_digraphes():
    assert translitterer_mot("7aja") == "حجا"
    assert translitterer_mot("3ijbitni")[0] == "ع"
    assert translitterer_mot("chbik").startswith("ش")


def test_exemple_de_la_demo():
    assert translitterer("ya bhim sakker fommek") == "يا بهيم سكر فمك"


def test_nombre_isole_et_arabe_conserves():
    assert translitterer("20 ans") == "20 انس"
    assert translitterer("برافو 3lik") == "برافو عليك"


def test_lettres_repetees_reduites():
    assert translitterer_mot("bhiiiiim") == translitterer_mot("bhiim")


def test_lexique_mot_entier_et_prefixe():
    lex = charger(["بهيم", "كلب"])
    assert termes_trouves("يا بهيم سكر فمك", lex) == ["بهيم"]
    assert termes_trouves("والكلب هذا", lex) == ["كلب"]
    assert predire("برافو عليك", lex) == "non_haineux"


def test_candidats_viennent_des_messages_haineux():
    textes = ["يا بهيم"] * 6 + ["برافو عليك"] * 6
    y = ["haineux"] * 6 + ["non_haineux"] * 6
    table = candidats(textes, y, min_haineux=5)
    assert "بهيم" in set(table.terme)
    assert "برافو" not in set(table.terme)
