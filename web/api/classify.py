"""Fonction Vercel : GET /api/classify?text=... -> trace JSON de la démo (même format que la démo locale)."""
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.modele import MAX_CHARS, Modele, classify  # noqa: E402

MODELE = Modele(Path(__file__).resolve().parent.parent / "model" / "modele_demo.npz")   # chargé une fois par instance


class handler(BaseHTTPRequestHandler):  # noqa: N801 (nom imposé par Vercel)
    def do_GET(self):  # noqa: N802
        texte = parse_qs(urlparse(self.path).query).get("text", [""])[0][: MAX_CHARS + 1]
        body = json.dumps(classify(texte, MODELE), ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
