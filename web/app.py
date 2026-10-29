"""Démo en ligne CrossHateTN : application WSGI sans dépendance autre que numpy (point d'entrée Vercel).

    GET /                       -> la page de la démo (index.html, identique à la démo locale)
    GET /api/classify?text=...  -> la trace JSON (même format que `python -m crosshatetn demo`)

Le modèle (web/model/modele_demo.npz) et le dossier demo_web/ sont produits par `python -m crosshatetn export-web`.
"""
import json
from pathlib import Path
from urllib.parse import parse_qs

from demo_web.modele import MAX_CHARS, Modele, classify

ICI = Path(__file__).resolve().parent
MODELE = Modele(ICI / "model" / "modele_demo.npz")   # chargé une fois par instance
PAGE = (ICI / "index.html").read_bytes()


def app(environ, start_response):
    chemin = environ.get("PATH_INFO", "/") or "/"
    if chemin in ("/", "/index.html"):
        body, ctype = PAGE, "text/html; charset=utf-8"
    elif chemin == "/api/classify":
        texte = parse_qs(environ.get("QUERY_STRING", "")).get("text", [""])[0][: MAX_CHARS + 1]
        body = json.dumps(classify(texte, MODELE), ensure_ascii=False).encode("utf-8")
        ctype = "application/json; charset=utf-8"
    else:
        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return ["Page introuvable".encode("utf-8")]
    start_response("200 OK", [("Content-Type", ctype), ("Content-Length", str(len(body))), ("Cache-Control", "no-store")])
    return [body]


if __name__ == "__main__":   # test local : python app.py puis http://127.0.0.1:8001
    from wsgiref.simple_server import make_server
    print("Démo web : http://127.0.0.1:8001")
    make_server("127.0.0.1", 8001, app).serve_forever()
