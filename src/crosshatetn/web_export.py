"""Prépare la démo en ligne (dossier web/, déployable sur Vercel).

    python -m crosshatetn export-web

1. entraîne le modèle de la démo exactement comme `demo` (T-HSAB train gelé, TF-IDF caractères, LinearSVC C = 1) ;
2. écrit web/model/modele_demo.npz : vocabulaire de n-grammes, idf, poids, biais (aucun texte du corpus) ;
3. copie text.py, translit.py et web_modele.py (inférence numpy) dans web/demo_web/ et la page de la démo dans web/index.html ;
4. vérifie que l'inférence numpy de web/ redonne les mêmes scores que scikit-learn, sinon s'arrête.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np

from . import demo

PHRASES_CONTROLE = ["يا بهيم سكر فمك", "قالي يا بهيم وضحكنا", "برافو عليك ربي يوفقك", "ya bhim sakker fommek",
                    "Bonjour, je vais bien merci à tous les amis", "n7eb tounes", "   ", "123 !!!",
                    "شكون بش يتفرج عليه غدوا هه", "مايا راكي تعيفي وتعييييييي", "ya 7mar  ya   bhim", "ﷲ يرحمو"]


def export(model: demo.DemoModel, web_dir: Path) -> Path:
    web_dir = Path(web_dir)
    lib = web_dir / "demo_web"
    (web_dir / "model").mkdir(parents=True, exist_ok=True)
    lib.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    for f, cible in (("text.py", "text.py"), ("translit.py", "translit.py"), ("web_modele.py", "modele.py")):
        shutil.copyfile(src / f, lib / cible)
    (lib / "__init__.py").touch()
    (web_dir / "index.html").write_text(demo.PAGE, encoding="utf-8")

    vec, svm = model.vectorizer, model.svm
    sign = demo._sign(model)
    out = web_dir / "model" / "modele_demo.npz"
    np.savez_compressed(out, vocab=np.asarray(model.names, dtype=str), idf=vec.idf_.astype(np.float64),
                        coef=(sign * svm.coef_[0]).astype(np.float64), intercept=np.float64(sign * svm.intercept_[0]),
                        ngram_range=np.asarray(vec.ngram_range), description=np.asarray(model.description + " Démo en ligne."))

    # Contrôle : l'inférence numpy de web/ doit redonner exactement la trace de la démo locale.
    sys.path.insert(0, str(web_dir))
    try:
        for name in list(sys.modules):
            if name == "demo_web" or name.startswith("demo_web."):
                del sys.modules[name]
        spec = importlib.util.find_spec("demo_web.modele")
        web = importlib.util.module_from_spec(spec)
        sys.modules["demo_web.modele"] = web
        spec.loader.exec_module(web)
        m = web.Modele(out)
        for phrase in PHRASES_CONTROLE:
            a, b = demo.classify(phrase, model), web.classify(phrase, m)
            if "ngrammes" in a:   # à poids égal, l'ordre de deux n-grammes peut différer : on compare les poids
                for t in (a, b):
                    ng = t["ngrammes"]
                    t["_poids"] = (ng["nombre"], [x["poids"] for x in ng["vers_haineux"]],
                                   [x["poids"] for x in ng["vers_non_haineux"]])
            for k in ("statut", "decision", "score", "_poids", "translitteration", "normalisation"):
                if a.get(k) != b.get(k):
                    raise AssertionError(f"Démo web différente de la démo locale pour {phrase!r} ({k}) : {a.get(k)} / {b.get(k)}")
    finally:
        sys.path.remove(str(web_dir))
    return out
