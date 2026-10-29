"""Écriture des sorties régénérables (outputs/) et du rapport de données en Markdown."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CORPUS_COLUMNS = ["id", "source", "source_row", "dialect", "role", "status", "label_source", "label", "label_status",
                  "split", "group", "dup_group", "n_in_group", "script", "arabizi_digits", "french_ratio", "n_chars",
                  "text_raw", "text_light", "text_strong"]
MODEL_COLUMNS = ["id", "source", "dialect", "script", "label", "label_status", "split", "group",
                 "text_raw", "text_light", "text_strong"]


def markdown_table(df: pd.DataFrame, index: bool = True) -> str:
    """Tableau Markdown sans dépendance externe."""
    frame = df.reset_index() if index else df
    cells = [[str(c) for c in frame.columns]] + [["" if pd.isna(v) else str(v) for v in row] for row in frame.to_numpy()]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "|".join("---" for _ in cells[0]) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in cells[1:]]
    return "\n".join(lines)


def write_outputs(prepared, out_dir: Path) -> None:
    data_dir, report_dir = Path(out_dir) / "data", Path(out_dir) / "rapports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    prepared.corpus[CORPUS_COLUMNS].to_csv(data_dir / "corpus.csv", index=False)
    prepared.labeled[MODEL_COLUMNS].to_csv(data_dir / "etiquete.csv", index=False)
    prepared.pool[["id", "source", "script", "arabizi_digits", "french_ratio", "label_source", "group",
                   "text_raw", "text_light"]].to_csv(data_dir / "reserve_arabizi.csv", index=False)
    prepared.conflicts.to_csv(data_dir / "conflits_a_arbitrer.csv", index=False)
    prepared.audit.review_pairs.to_csv(data_dir / "quasi_doublons_a_revoir.csv", index=False)
    prepared.audit.table().to_csv(report_dir / "audit.csv", index=False)
    (report_dir / "provenance.json").write_text(json.dumps(prepared.provenance, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    (report_dir / "rapport_donnees.md").write_text(render_report(prepared), encoding="utf-8")


def render_report(prepared) -> str:
    """Rapport factuel régénéré à chaque exécution (base des chapitres données du rapport)."""
    cfg, corpus, labeled = prepared.cfg, prepared.corpus, prepared.labeled
    nd, split_cfg = cfg.section("near_duplicates"), cfg.section("split")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prov = pd.DataFrame(prepared.provenance).T[["role", "dialect", "rows", "commit", "sha256"]]
    prov["commit"] = prov["commit"].str[:10]
    prov["sha256"] = prov["sha256"].str[:16] + "…"
    prov.index.name = "source"
    lengths = (corpus[corpus["status"] == "ok"].groupby("source")["n_chars"]
               .describe()[["mean", "50%", "min", "max"]].round(1)
               .rename(columns={"mean": "moyenne", "50%": "médiane"}))
    lengths[["min", "max"]] = lengths[["min", "max"]].astype(int)
    sizes = labeled["group"].value_counts()
    pool = prepared.pool
    parts = [
        "# Rapport de données CrossHateTN",
        f"Généré le {now}. Ne pas modifier à la main : relancer `python -m crosshatetn prepare`.",
        "## Sources (versions figées)",
        markdown_table(prov),
        "## Devenir des lignes",
        markdown_table(prepared.summary()),
        f"Doublons : même clé normalisée dans une même source. Conflits : politique « {cfg.section('dedup')['conflict_policy']} » "
        f"({len(prepared.conflicts)} groupe(s), à arbitrer dans `{cfg.data['paths']['arbitration']}`).",
        "## Systèmes d'écriture (textes conservés)",
        markdown_table(prepared.script_table()),
        "## Longueur des textes conservés (caractères)",
        markdown_table(lengths),
        "## Splits (textes étiquetés conservés)",
        f"Stratifiés sur {' × '.join(split_cfg['stratify_on'])}, groupés par quasi-doublons "
        f"(cosinus ≥ {nd['group_threshold']} sur n-grammes de caractères {tuple(nd['ngram_range'])}), "
        f"graine {cfg.seed}. Groupes de 2 textes ou plus : {int((sizes > 1).sum())} (le plus grand : {int(sizes.max())}).",
        markdown_table(prepared.split_table()),
        "## Audit des fuites",
        markdown_table(prepared.audit.table(), index=False),
        "## Réserve Arabizi (TUNIZI)",
        f"{len(pool)} phrases conservées ; {int(pool['arabizi_digits'].sum())} contiennent des chiffres-lettres "
        f"(3, 7, 9…) ; {int((pool['french_ratio'] >= 0.3).sum())} ressemblent plutôt à du français "
        "(≥ 30 % de mots outils français). Étiquettes d'origine : sentiment, pas haine — réannotation nécessaire.",
    ]
    return "\n\n".join(parts) + "\n"
