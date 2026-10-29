import tempfile
from pathlib import Path

import pandas as pd

from crosshatetn.demo import fit_model
from crosshatetn.web_export import export


def test_demo_web_identique_a_la_demo_locale():
    hate = ["يا بهيم سكر فمك", "انت بهيم يا حمار", "اسكت يا بهيم", "يا حمار جاهل", "بهيم وحمار"] * 4
    ok = ["برافو عليك ربي يوفقك", "ربي يهديكم جميعا", "مبروك النجاح يا صاحبي", "الله يعطيك الصحة"] * 4
    train = pd.DataFrame({"text_raw": hate + ok, "label": ["abusive"] * len(hate) + ["normal"] * len(ok)})
    with tempfile.TemporaryDirectory() as d:
        # export() s'arrête si l'inférence numpy de web/ ne redonne pas les scores de scikit-learn
        out = export(fit_model(train, seed=42, description="test"), Path(d))
        assert out.exists() and (Path(d) / "index.html").exists()
        assert (Path(d) / "demo_web" / "modele.py").exists()
