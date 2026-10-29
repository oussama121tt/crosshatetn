"""Récupération figée des sources (commit + SHA-256) et conversion vers le schéma commun.

Les textes ne sont jamais versionnés : ils sont retéléchargés à une version figée,
puis vérifiés par empreinte avant toute utilisation.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

SCHEMA = ["id", "source", "source_row", "dialect", "role", "text_raw", "label_source", "label"]


class SourceError(RuntimeError):
    """Source introuvable, altérée ou illisible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, timeout: int = 60) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _git_fetch_file(repo: str, commit: str, rel_path: str, dest: Path) -> None:
    """Récupère un fichier à un commit précis sans cloner tout l'historique."""
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        def git(*args: str) -> None:
            subprocess.run(["git", "-C", tmp, *args], check=True, capture_output=True, timeout=300)

        git("init", "-q")
        git("remote", "add", "origin", f"https://github.com/{repo}.git")
        git("fetch", "-q", "--depth", "1", "origin", commit)
        git("checkout", "-q", "FETCH_HEAD")
        shutil.copyfile(Path(tmp) / rel_path, dest)


def fetch(name: str, spec: dict, raw_dir: Path) -> Path:
    """Renvoie le chemin local de la source, téléchargée si besoin, et vérifiée par SHA-256."""
    dest = raw_dir / name / Path(spec["path"]).name
    if dest.exists() and sha256_file(dest) == spec["sha256"]:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/"
           + urllib.parse.quote(spec["path"]))
    errors = []
    for method, action in (("https", lambda: _download(url, dest)),
                           ("git", lambda: _git_fetch_file(spec["repo"], spec["commit"], spec["path"], dest))):
        try:
            action()
            break
        except Exception as exc:  # on essaie la méthode suivante
            errors.append(f"{method} : {exc}")
    else:
        raise SourceError(f"Impossible de récupérer la source '{name}'. Vérifie l'accès à Internet "
                          f"(sur Kaggle : Internet activé). Détails : {' | '.join(errors)}")
    found = sha256_file(dest)
    if found != spec["sha256"]:
        dest.unlink(missing_ok=True)
        raise SourceError(f"Empreinte inattendue pour '{name}' : {found} au lieu de {spec['sha256']}. "
                          "Le fichier a changé : ne pas continuer sans revoir le protocole.")
    return dest


# --- Lecture des formats d'origine -------------------------------------------------------------

def read_thsab_xlsx(path: Path) -> pd.DataFrame:
    """T-HSAB : Excel sans en-tête, colonne 1 = texte, colonne 2 = étiquette."""
    df = pd.read_excel(path, header=None, dtype=object)
    if df.shape[1] != 2:
        raise SourceError(f"T-HSAB : 2 colonnes attendues, {df.shape[1]} trouvées")
    df.columns = ["text_raw", "label_source"]
    return df


def read_lhsab_tsv(path: Path) -> pd.DataFrame:
    """L-HSAB : TSV avec en-tête 'Tweet', 'Class' ; certains tweets entre guillemets tiennent sur plusieurs lignes."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if list(df.columns) != ["Tweet", "Class"]:
        raise SourceError(f"L-HSAB : colonnes inattendues {list(df.columns)}")
    return df.rename(columns={"Tweet": "text_raw", "Class": "label_source"})


# Une ligne TUNIZI = "<1|-1>;<phrase>". Quelques lignes ont un caractère parasite devant l'étiquette
# (ex. '%1;' ou '"1;') : on l'ignore, sans jamais avaler le signe moins.
TUNIZI_LINE = re.compile(r"^[^\w-]*(-?1)\s*;(.*)$")


def read_tunizi_txt(path: Path) -> pd.DataFrame:
    rows, bad = [], []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        match = TUNIZI_LINE.match(line)
        if match is None:
            bad.append(lineno)
        else:
            rows.append((match.group(2), match.group(1)))
    if bad:
        raise SourceError(f"TUNIZI : lignes illisibles {bad[:10]}{'…' if len(bad) > 10 else ''}")
    return pd.DataFrame(rows, columns=["text_raw", "label_source"])


READERS = {"thsab_xlsx": read_thsab_xlsx, "lhsab_tsv": read_lhsab_tsv, "tunizi_txt": read_tunizi_txt}


def to_schema(name: str, spec: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Convertit une source vers le schéma commun. Les identifiants sont stables (rang dans le fichier figé)."""
    n = len(df)
    label_source = df["label_source"].map(lambda v: "" if pd.isna(v) else str(v).strip())
    out = pd.DataFrame({
        "id": [f"{name}-{i:05d}" for i in range(1, n + 1)],
        "source": name,
        "source_row": np.arange(1, n + 1),
        "dialect": spec["dialect"],
        "role": spec["role"],
        "text_raw": df["text_raw"].map(lambda v: "" if pd.isna(v) else str(v)),
        "label_source": label_source,
    })
    if spec["role"] == "labeled":
        mapped = out["label_source"].map(spec["label_map"])
        unknown = sorted(out.loc[mapped.isna(), "label_source"].unique())
        if unknown:
            raise SourceError(f"Source '{name}' : étiquettes non prévues dans label_map : {unknown}")
        out["label"] = mapped.astype(object)
    else:
        out["label"] = None
    return out[SCHEMA]


def load_all(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Charge toutes les sources de la configuration. Renvoie le corpus unifié et la provenance."""
    raw_dir = cfg.path("raw")
    frames, provenance = [], {}
    for name, spec in cfg.sources.items():
        path = fetch(name, spec, raw_dir)
        unified = to_schema(name, spec, READERS[spec["loader"]](path))
        frames.append(unified)
        provenance[name] = {
            "repo": f"https://github.com/{spec['repo']}",
            "commit": spec["commit"],
            "file": spec["path"],
            "sha256": spec["sha256"],
            "role": spec["role"],
            "dialect": spec["dialect"],
            "rows": len(unified),
        }
    corpus = pd.concat(frames, ignore_index=True)
    if corpus["id"].duplicated().any():
        raise SourceError("Identifiants en double après fusion des sources")
    return corpus, provenance
