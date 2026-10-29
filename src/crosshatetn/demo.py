"""Démonstration de bout en bout (palier Bronze).

Entrée libre → écriture détectée → translittération par règles si Arabizi → n-grammes pondérés →
score du SVM → classe, avec la trace de chaque étape et la latence mesurée.
Composants classiques uniquement (règles, TF-IDF caractères, SVM linéaire) : aucun LLM, aucune API.

La sortie est une aide à la revue humaine, jamais une sanction automatique.
Lancement : python -m crosshatetn demo [--port 8000]
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .config import Config
from .pipeline import prepare
from .text import detect_script, french_ratio, has_arabizi_digits, normalize_light, normalize_strong
from .translit import translitterer

HATE, NON_HATE = "haineux", "non_haineux"
BINARY = {"normal": NON_HATE, "abusive": HATE, "hate": HATE}
MAX_CHARS = 500        # au-delà, la démo refuse le texte (hors cadre pour une démonstration)
TOP_K = 6              # n-grammes affichés dans chaque sens
NEAR_ZERO = 0.1        # |score| inférieur : signalé « proche de la frontière », la classe ne change pas
FRENCH_LIMIT = 0.2     # part de mots français au-delà de laquelle un texte latin est hors périmètre


@dataclass
class DemoModel:
    vectorizer: TfidfVectorizer
    svm: LinearSVC
    names: np.ndarray
    n_train: int
    description: str


def fit_model(train: pd.DataFrame, seed: int = 42, description: str = "") -> DemoModel:
    """Entraîne la référence : TF-IDF caractères (2 à 5, char_wb) + LinearSVC, C = 1, classes équilibrées.

    train : colonnes text_raw et label (normal / abusive / hate), 2 classes après regroupement.
    """
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X = vectorizer.fit_transform(train["text_raw"])
    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=seed).fit(X, train["label"].map(BINARY))
    return DemoModel(vectorizer, svm, vectorizer.get_feature_names_out(), len(train), description)


def build_model(cfg: Config) -> DemoModel:
    """Données gelées : seuls les commentaires T-HSAB d'entraînement (4 164) servent à entraîner."""
    lab = prepare(cfg, write=False, verbose=False).labeled
    train = lab[(lab["source"] == "thsab") & (lab["split"] == "train")]
    description = (f"T-HSAB, entraînement gelé : {len(train)} commentaires ; TF-IDF caractères 2 à 5 ; "
                   "LinearSVC, C = 1, classes équilibrées ; 2 classes ; Arabizi translittéré par règles.")
    return fit_model(train, seed=cfg.seed, description=description)


def check_input(text: str) -> tuple[str, str]:
    """Cas d'échec gérés avant tout calcul : (statut, message). Statut « ok » si le texte est acceptable."""
    stripped = text.strip()
    if not stripped:
        return "vide", "Entrée vide : rien à classer."
    if len(stripped) > MAX_CHARS:
        return "trop_long", f"Plus de {MAX_CHARS} caractères : la démo refuse les textes longs."
    if not any(ch.isalpha() for ch in stripped):
        return "hors_perimetre", "Aucune lettre : émojis, chiffres ou ponctuation seuls."
    script = detect_script(stripped)
    if script == "other":
        return "hors_perimetre", "Alphabet non pris en charge : le modèle lit l'arabe et l'Arabizi latin."
    if script == "latin" and not has_arabizi_digits(stripped) and french_ratio(stripped) >= FRENCH_LIMIT:
        return "hors_perimetre", "Texte probablement en français : le modèle est entraîné sur le tunisien."
    return "ok", ""


def _sign(model: DemoModel) -> float:
    """+1 si la décision positive désigne « haineux », -1 sinon (ordre alphabétique des classes)."""
    return 1.0 if model.svm.classes_[1] == HATE else -1.0


def _run(text: str, model: DemoModel) -> dict:
    """Score et trace d'un texte déjà accepté : n-grammes, poids et score par classe."""
    X = model.vectorizer.transform([text.strip()])
    sign = _sign(model)
    score = sign * float(model.svm.decision_function(X)[0])
    weights = sign * model.svm.coef_[0]
    row = X.tocsr()
    contrib = weights[row.indices] * row.data
    order = np.argsort(contrib)
    vers_haineux = [(model.names[row.indices[i]], float(contrib[i])) for i in order[::-1] if contrib[i] > 0][:TOP_K]
    vers_non = [(model.names[row.indices[i]], float(contrib[i])) for i in order if contrib[i] < 0][:TOP_K]
    return {
        "score": round(score, 3),
        "decision": HATE if score > 0 else NON_HATE,
        "proche_frontiere": abs(score) < NEAR_ZERO,
        "ngrammes": {
            "nombre": int(row.nnz),
            "vers_haineux": [{"ngramme": g, "poids": round(w, 3)} for g, w in vers_haineux],
            "vers_non_haineux": [{"ngramme": g, "poids": round(w, 3)} for g, w in vers_non],
        },
    }


def classify(text: str, model: DemoModel) -> dict:
    """Trace complète d'une entrée. Latence = translittération + vectorisation + score + lecture des n-grammes."""
    t0 = time.perf_counter()
    status, message = check_input(text)
    trace: dict = {"entree": text, "statut": status, "message": message}
    if status != "ok":
        trace["decision"] = "abstention"
    else:
        stripped = text.strip()
        script = detect_script(stripped)
        lu = translitterer(stripped) if script in ("latin", "mixed") else stripped
        trace["ecriture"] = script
        trace["arabizi_chiffres"] = has_arabizi_digits(stripped)
        # Les normalisations sont montrées sur le texte réellement lu (testées en E3, non retenues).
        trace["normalisation"] = {"brut": stripped, "legere": normalize_light(lu),
                                  "forte": normalize_strong(lu),
                                  "utilise_par_le_modele": "brut" if lu == stripped else "translittéré"}
        trace.update(_run(lu, model))
        if lu != stripped:
            sans = _run(stripped, model)
            trace["translitteration"] = {"statut": "regles", "arabe": lu,
                                         "score_sans": sans["score"], "decision_sans": sans["decision"]}
    trace["latence_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    trace["modele"] = model.description
    trace["avertissement"] = "Aide à la revue humaine : jamais une sanction automatique."
    return trace


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CrossHateTN : démonstration</title>
<style>
/* Palette ESPRIM, comme les diapos : bleu #0B2B7C / #173888, rouge #EE1C25 */
:root{--ink:#1D2B53;--blue:#0B2B7C;--hblue:#173888;--red:#EE1C25;--line:#E2E8F3;--bg:#F4F6FB;--card:#FFFFFF;--muted:#5A6785;
      --tblue:#EEF2FA;--tred:#FFF1F2;--ok:#1E9E5A}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#E8ECF6;--blue:#9FB4F2;--hblue:#22409A;--red:#FF4B52;--line:#2C3550;--bg:#0E1424;--card:#161E33;--muted:#A4AFC9;--tblue:#1C2744;--tred:#3A1A22;--ok:#4CC789}}
:root[data-theme="dark"]{--ink:#E8ECF6;--blue:#9FB4F2;--hblue:#22409A;--red:#FF4B52;--line:#2C3550;--bg:#0E1424;--card:#161E33;--muted:#A4AFC9;--tblue:#1C2744;--tred:#3A1A22;--ok:#4CC789}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI",system-ui,Arial,sans-serif;padding:16px}
main{max-width:1040px;margin:0 auto}
header{display:flex;align-items:center;gap:16px;margin:6px 0 4px}
.badge{flex:none;width:58px;height:58px;border-radius:14px;background:var(--red);color:#FFF;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:24px}
h1{margin:0;font-size:30px;line-height:1.1;color:var(--blue)} h1 span{color:var(--red)}
.trait{width:120px;height:6px;border-radius:3px;background:linear-gradient(90deg,var(--red) 50%,var(--blue) 50%);margin:10px 0 6px 74px}
.muted{color:var(--muted);font-size:14px;margin:4px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;margin:14px 0;overflow:hidden}
.card>.head{display:flex;align-items:center;gap:10px;background:var(--hblue);color:#FFF;padding:10px 16px;font-weight:600;font-size:17px}
.card>.head.red{background:var(--red)}
.card>.body{padding:14px 16px}
.num{flex:none;width:28px;height:28px;border-radius:14px;background:#FFF;color:var(--hblue);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:15px}
.head.red .num{color:var(--red)}
textarea{width:100%;min-height:76px;font-size:21px;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--card);color:var(--ink);font-family:inherit;resize:vertical}
textarea:focus{outline:2px solid var(--hblue);outline-offset:1px}
.boutons{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
button{font-size:15px;padding:9px 14px;border-radius:10px;border:1px solid var(--line);background:var(--tblue);color:var(--ink);cursor:pointer;font-family:inherit}
button:hover{border-color:var(--hblue)}
button.primary{background:var(--red);color:#FFF;border-color:var(--red);font-weight:700;padding:9px 22px}
.verdict{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;padding:16px 20px;border-radius:16px;margin:14px 0;font-weight:700;font-size:22px}
.verdict.haineux{background:var(--red);color:#FFF}
.verdict.non{background:var(--ok);color:#FFF}
.verdict.abst{background:var(--card);border:2px dashed var(--muted);color:var(--ink);font-size:19px}
.verdict small{font-weight:500;font-size:15px;opacity:.95}
.jauge{position:relative;height:14px;border-radius:7px;background:linear-gradient(90deg,var(--ok),#D7DCE8 50%,var(--red));margin:8px 0 4px}
.jauge i{position:absolute;top:-5px;width:4px;height:24px;border-radius:2px;background:var(--ink)}
.jauge-l{display:flex;justify-content:space-between;font-size:13px;color:var(--muted)}
.etapes{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
@media (max-width:820px){.etapes{grid-template-columns:1fr}}
.etape{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.etape .t{background:var(--hblue);color:#FFF;font-weight:600;font-size:15px;padding:8px 10px;text-align:center}
.etape:nth-child(even) .t{background:var(--red)}
.etape .v{padding:10px;font-size:15px;display:flex;flex-direction:column;gap:6px;flex:1;justify-content:center;text-align:center}
.etape .v b{color:var(--blue)}
.ar{direction:rtl;unicode-bidi:isolate;font-size:22px;font-family:"Noto Sans Arabic","Segoe UI",Arial,sans-serif}
table{width:100%;border-collapse:collapse;font-size:15px}
td,th{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:middle}
th{width:38%;color:var(--blue);font-weight:600}
tr.use td,tr.use th{background:var(--tred)}
.tag{display:inline-block;font-size:12px;font-weight:700;color:#FFF;background:var(--red);border-radius:6px;padding:2px 6px;margin-left:6px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:700px){.cols{grid-template-columns:1fr}}
.ng{display:flex;align-items:center;gap:10px;margin:6px 0}
.ng .g{flex:none;min-width:84px;text-align:center;background:var(--tblue);border-radius:8px;padding:3px 8px}
.ng .bar{height:12px;border-radius:6px}
.ng .w{font-size:14px;color:var(--muted);font-variant-numeric:tabular-nums}
.pied{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:14px;color:var(--muted)}
</style></head>
<body><main>
<header><div class="badge">03</div><h1>CrossHateTN : <span>de la phrase à la classe</span></h1></header>
<div class="trait"></div>
<p class="muted">Tapez une phrase en arabe tunisien ou en Arabizi, puis « Classer » (ou Ctrl + Entrée). Chaque étape reste visible.</p>
<div class="card"><div class="head"><span class="num">✎</span>Entrée</div><div class="body">
  <textarea id="entree" dir="auto" placeholder="Tapez une phrase…"></textarea>
  <div class="boutons">
    <button class="primary" id="go">Classer</button>
    <button data-ex="يا بهيم سكر فمك">Insulte directe</button>
    <button data-ex="قالي يا بهيم وضحكنا">Insulte citée</button>
    <button data-ex="برافو عليك ربي يوفقك">Encouragement</button>
    <button data-ex="ya bhim sakker fommek">Arabizi</button>
    <button data-ex="Bonjour, je vais bien merci à tous les amis">Français (hors périmètre)</button>
  </div>
</div></div>
<div id="sortie"></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt !== undefined) e.textContent = txt; return e; };
const nb = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d).replace(".", ",");
const ECRITURE = { arabic: "Arabe", latin: "Arabizi (lettres latines)", mixed: "Mixte (arabe + latin)" };
function card(titre, n, rouge) {
  const c = el("div", "card"); const h = el("div", "head" + (rouge ? " red" : ""));
  h.append(el("span", "num", n), document.createTextNode(titre)); const b = el("div", "body"); c.append(h, b); return [c, b];
}
function etape(titre, contenu) { const e = el("div", "etape"); e.append(el("div", "t", titre)); const v = el("div", "v"); contenu(v); e.append(v); return e; }
function ligne(tbl, label, valeur, ar, use) {
  const tr = el("tr", use ? "use" : ""); const th = el("th", "", label); if (use) th.append(el("span", "tag", "lu par le modèle"));
  const td = el("td", ar ? "ar" : ""); td.textContent = valeur; tr.append(th, td); tbl.append(tr);
}
function ngrammes(box, titre, items, couleur) {
  box.append(el("p", "", titre));
  if (!items.length) { box.append(el("p", "muted", "aucun morceau dans ce sens")); return; }
  const max = Math.max(...items.map(i => Math.abs(i.poids)));
  items.forEach(it => {
    const r = el("div", "ng"); const g = el("span", "g ar", it.ngramme.replace(/ /g, "·")); g.style.fontSize = "19px";
    const b = el("div", "bar"); b.style.width = (12 + 160 * Math.abs(it.poids) / max) + "px"; b.style.background = couleur;
    r.append(g, b, el("span", "w", nb(it.poids))); box.append(r);
  });
}
function rendu(t) {
  const out = $("sortie"); out.innerHTML = "";
  if (t.statut !== "ok") {
    const v = el("div", "verdict abst"); v.append(el("span", "", "Abstention : " + t.message), el("small", "", "Aucun score calculé.")); out.append(v); return;
  }
  const h = t.decision === "haineux";
  const v = el("div", "verdict " + (h ? "haineux" : "non"));
  v.append(el("span", "", (h ? "Haineux : à relire" : "Non haineux") + " · score " + nb(t.score)),
           el("small", "", t.proche_frontiere ? "Score proche de 0 : décision fragile" : "Score > 0 → haineux · score < 0 → non haineux"));
  out.append(v);
  const pos = Math.max(2, Math.min(98, 50 + 50 * t.score / 1.5));
  const jg = el("div", "jauge"); const m = el("i"); m.style.left = "calc(" + pos + "% - 2px)"; jg.append(m);
  const jl = el("div", "jauge-l"); jl.append(el("span", "", "non haineux"), el("span", "", "0"), el("span", "", "haineux"));
  const lu = t.translitteration ? t.translitteration.arabe : t.normalisation.brut;
  const et = el("div", "etapes");
  et.append(
    etape("1. Entrée", v => { v.append(el("b", "", ECRITURE[t.ecriture] || t.ecriture)); v.append(el("span", "muted", t.arabizi_chiffres ? "chiffres Arabizi (3, 7, 9…) repérés" : "écriture acceptée")); }),
    etape("2. Prétraitement", v => { v.append(el("b", "", t.translitteration ? "Translittération par règles" : "Texte brut")); v.append(el("span", "ar", lu)); }),
    etape("3. Représentation", v => { v.append(el("b", "", "TF-IDF caractères")); v.append(el("span", "muted", t.ngrammes.nombre + " morceaux de 2 à 5 lettres")); }),
    etape("4. Classification", v => { v.append(el("b", "", "SVM linéaire")); v.append(el("span", "muted", "somme des poids = " + nb(t.score))); }),
    etape("5. Sortie", v => { v.append(el("b", "", h ? "Haineux" : "Non haineux")); v.append(el("span", "muted", t.latence_ms.toFixed(2).replace(".", ",") + " ms")); })
  );
  const [cj, bj] = card("Score du SVM", "Σ"); bj.append(jg, jl); out.append(cj);
  out.append(et);
  const [c2, b2] = card("Texte lu par le modèle", "T", true);
  const tb = el("table");
  ligne(tb, "Entrée", t.normalisation.brut, t.ecriture !== "latin", !t.translitteration);
  if (t.translitteration) ligne(tb, "Après translittération", t.translitteration.arabe, true, true);
  ligne(tb, "Normalisation légère (testée, non retenue)", t.normalisation.legere, true, false);
  ligne(tb, "Normalisation forte (testée, non retenue)", t.normalisation.forte, true, false);
  b2.append(tb);
  if (t.translitteration) b2.append(el("p", "muted", "Sans translittération, l'Arabizi brut donnerait : score " + nb(t.translitteration.score_sans) + " (" + (t.translitteration.decision_sans === "haineux" ? "haineux" : "non haineux") + ")."));
  out.append(c2);
  const [c3, b3] = card("Morceaux qui pèsent le plus (" + t.ngrammes.nombre + " actifs) · « · » = début ou fin de mot", "N");
  const cols = el("div", "cols"); const a = el("div"); const b = el("div");
  ngrammes(a, "Vers haineux", t.ngrammes.vers_haineux, "var(--red)");
  ngrammes(b, "Vers non haineux", t.ngrammes.vers_non_haineux, "var(--ok)");
  cols.append(a, b); b3.append(cols); out.append(c3);
  const p = el("div", "pied"); p.append(el("span", "", "Latence : " + t.latence_ms.toFixed(2).replace(".", ",") + " ms"), el("span", "", t.modele), el("span", "", t.avertissement));
  out.append(p);
}
async function classer(texte) {
  $("entree").value = texte;
  const r = await fetch("/api/classify?text=" + encodeURIComponent(texte));
  rendu(await r.json());
}
$("go").onclick = () => classer($("entree").value);
$("entree").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); classer($("entree").value); } });
document.querySelectorAll("button[data-ex]").forEach(b => b.onclick = () => classer(b.dataset.ex));
</script></body></html>
"""


def make_handler(model: DemoModel) -> type:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (nom imposé par http.server)
            url = urlparse(self.path)
            if url.path == "/":
                body, ctype = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            elif url.path == "/api/classify":
                text = parse_qs(url.query).get("text", [""])[0][: MAX_CHARS + 1]
                body = json.dumps(classify(text, model), ensure_ascii=False).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # silencieux : la trace est dans la page
            return

    return Handler


def serve(model: DemoModel, port: int = 8000) -> None:
    """Écoute sur 127.0.0.1 uniquement."""
    classify("يا بهيم سكر فمك", model)          # premier appel à vide : la latence affichée ne compte pas le démarrage
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(model))
    print(f"Démo prête : http://127.0.0.1:{port}  (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
