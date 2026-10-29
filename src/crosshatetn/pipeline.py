"""Orchestration du socle de données.

sources figées → schéma commun → normalisation → doublons et conflits → groupes de quasi-doublons
→ splits gelés → audit des fuites → sorties.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .audit import AuditResult, LeakageError, run_audit
from .config import Config, load_config
from .dedup import mark_duplicates, near_duplicate_groups, read_arbitration, resolve_conflicts
from .report import write_outputs
from .sources import load_all
from .splits import (SplitOutcome, assign, check_frozen_digests, load_frozen, load_manifest, save_frozen)
from .text import add_text_features

STATUS_ORDER = ["ok", "doublon", "conflit", "vide"]


@dataclass
class Prepared:
    cfg: Config
    corpus: pd.DataFrame        # toutes les lignes de toutes les sources, avec leur statut
    labeled: pd.DataFrame       # textes étiquetés conservés, avec groupe et split : entrée des modèles
    pool: pd.DataFrame          # réserve Arabizi dédoublonnée (à annoter)
    conflicts: pd.DataFrame     # groupes aux étiquettes contradictoires
    pairs: pd.DataFrame         # paires de quasi-doublons (similarité >= seuil de revue)
    audit: AuditResult
    split_outcome: SplitOutcome
    provenance: dict

    def summary(self) -> pd.DataFrame:
        """Devenir des lignes de chaque source (brut → conservé)."""
        table = pd.crosstab(self.corpus["source"], self.corpus["status"]).reindex(columns=STATUS_ORDER, fill_value=0)
        table.insert(0, "lignes", table.sum(axis=1))
        return table.rename(columns={"ok": "conservées", "doublon": "doublons", "conflit": "conflits exclus",
                                     "vide": "vides"})

    def split_table(self) -> pd.DataFrame:
        """Effectifs par source, split et classe."""
        return (pd.crosstab([self.labeled["source"], self.labeled["split"]], self.labeled["label"], margins=True,
                            margins_name="total")
                .reindex(columns=self.cfg.labels + ["total"]))

    def script_table(self) -> pd.DataFrame:
        kept = self.corpus[self.corpus["status"] == "ok"]
        return pd.crosstab(kept["source"], kept["script"])


def data_fingerprint(cfg: Config) -> str:
    """Empreinte des seuls paramètres qui déterminent les données et les splits."""
    relevant = {
        "seed": cfg.seed,
        "sources": {k: v["sha256"] for k, v in cfg.sources.items()},
        "dedup": cfg.section("dedup"),
        "near_duplicates": cfg.section("near_duplicates"),
        "split": cfg.section("split"),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


def prepare(config: str | Path | Config = "configs/data.yaml", refreeze: bool = False, strict: bool = True,
            write: bool = True, verbose: bool = True) -> Prepared:
    """Construit le socle de données. Avec strict=True, toute fuite détectée arrête tout."""
    cfg = config if isinstance(config, Config) else load_config(config)
    log = print if verbose else (lambda *a, **k: None)

    corpus, provenance = load_all(cfg)
    loaded = ", ".join(f"{name} ({info['rows']})" for name, info in provenance.items())
    log(f"Sources chargées et vérifiées (SHA-256) : {loaded}")
    corpus = add_text_features(corpus)
    corpus = mark_duplicates(corpus)
    arbitration = read_arbitration(cfg.path("arbitration"), cfg.labels)
    corpus, conflicts = resolve_conflicts(corpus, arbitration, cfg.section("dedup")["conflict_policy"])

    units = corpus[corpus["status"].isin(["ok", "conflit"])]
    nd = cfg.section("near_duplicates")
    group, pairs = near_duplicate_groups(units, nd)
    corpus["group"] = corpus["id"].map(group)
    is_dup = corpus["status"] == "doublon"
    corpus.loc[is_dup, "group"] = corpus.loc[is_dup, "dup_group"].map(group)

    labeled = corpus[(corpus["role"] == "labeled") & (corpus["status"] == "ok")].copy()
    split_dir = cfg.path("splits")
    manifest = None if refreeze else load_manifest(split_dir)
    frozen = None if refreeze else load_frozen(split_dir)
    if frozen is not None:
        check_frozen_digests(frozen, manifest)
        if manifest and manifest.get("empreinte_donnees") not in (None, data_fingerprint(cfg)):
            log("Attention : la configuration des données a changé depuis le gel des splits. "
                "Les affectations existantes sont conservées ; re-geler seulement si c'est voulu.")
    outcome = assign(labeled[["id", "group", "source", "label"]], frozen, cfg.section("split"), cfg.seed, refreeze)
    labeled["split"] = labeled["id"].map(outcome.assignment)
    corpus["split"] = corpus["id"].map(outcome.assignment)
    if outcome.created:
        log(f"Splits créés : {outcome.assignment.value_counts().to_dict()}")
    else:
        log(f"Splits gelés chargés depuis {split_dir}"
            + (f" ; {len(outcome.added)} nouvel(s) identifiant(s) ajouté(s)" if len(outcome.added) else ""))

    audit = run_audit(labeled, pairs, nd["group_threshold"], nd["review_threshold"],
                      cfg.section("split")["max_class_gap"])
    pool = corpus[(corpus["role"] == "pool") & (corpus["status"] == "ok")].copy()
    prepared = Prepared(cfg, corpus, labeled, pool, conflicts, pairs, audit, outcome, provenance)

    if strict and not audit.ok:
        raise LeakageError("Fuite détectée entre splits :\n" + audit.table().to_string(index=False))
    log("Audit des fuites : " + ("aucune fuite détectée" if audit.ok else "ÉCHEC"))

    if write:
        if outcome.created or len(outcome.added):
            extra = {"empreinte_donnees": data_fingerprint(cfg), "graine": cfg.seed,
                     "sources_sha256": {k: v["sha256"] for k, v in cfg.sources.items()}}
            save_frozen(split_dir, outcome, frozen, labeled, extra)
            log(f"Affectation gelée écrite dans {split_dir} : à versionner (git add splits/)")
        write_outputs(prepared, cfg.path("outputs"))
        log(f"Sorties écrites dans {cfg.path('outputs')}")
    return prepared
