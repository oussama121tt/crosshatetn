"""Chargement et validation de la configuration YAML."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_KEYS = ("seed", "labels", "target_source", "paths", "sources", "dedup", "near_duplicates", "split")
LOADERS = ("thsab_xlsx", "lhsab_tsv", "tunizi_txt")
ROLES = ("labeled", "pool")


class ConfigError(ValueError):
    """Configuration invalide."""


@dataclass(frozen=True)
class Config:
    data: dict
    root: Path

    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def labels(self) -> list[str]:
        return list(self.data["labels"])

    @property
    def target_source(self) -> str:
        return self.data["target_source"]

    @property
    def sources(self) -> dict:
        return self.data["sources"]

    def path(self, key: str) -> Path:
        return (self.root / self.data["paths"][key]).resolve()

    def section(self, name: str) -> dict:
        return self.data[name]

    @property
    def fingerprint(self) -> str:
        """Empreinte courte de la configuration, enregistrée dans les manifestes."""
        canonical = json.dumps(self.data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def load_config(path: str | Path = "configs/data.yaml", root: str | Path | None = None) -> Config:
    """Charge la configuration. Par défaut, la racine du dépôt est le parent du dossier configs/."""
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"Configuration introuvable : {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = Config(data=data, root=Path(root).resolve() if root else path.parent.parent)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in cfg.data]
    if missing:
        raise ConfigError(f"Clés manquantes dans la configuration : {missing}")
    if cfg.target_source not in cfg.sources:
        raise ConfigError(f"target_source inconnue : {cfg.target_source}")
    for name, spec in cfg.sources.items():
        for key in ("role", "dialect", "loader", "repo", "commit", "path", "sha256"):
            if key not in spec:
                raise ConfigError(f"Source {name} : clé '{key}' manquante")
        if spec["role"] not in ROLES:
            raise ConfigError(f"Source {name} : rôle '{spec['role']}' inconnu (attendu : {ROLES})")
        if spec["loader"] not in LOADERS:
            raise ConfigError(f"Source {name} : chargeur '{spec['loader']}' inconnu (attendu : {LOADERS})")
        if spec["role"] == "labeled":
            targets = set(spec.get("label_map", {}).values())
            if not targets or not targets <= set(cfg.labels):
                raise ConfigError(f"Source {name} : label_map doit viser uniquement {cfg.labels}")
    split = cfg.section("split")
    if not 0 < split["test"] + split["validation"] < 1:
        raise ConfigError("Les proportions test + validation doivent être comprises entre 0 et 1")
    if cfg.section("dedup")["conflict_policy"] not in ("exclude", "majority"):
        raise ConfigError("dedup.conflict_policy doit valoir 'exclude' ou 'majority'")
    nd = cfg.section("near_duplicates")
    if not 0 < nd["review_threshold"] <= nd["group_threshold"] <= 1:
        raise ConfigError("Il faut 0 < review_threshold <= group_threshold <= 1")
