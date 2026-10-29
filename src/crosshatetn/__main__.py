"""Interface en ligne de commande.

    python -m crosshatetn prepare              # socle de données + splits gelés + audit
    python -m crosshatetn baseline             # E1 sur la validation (le test reste gelé)
    python -m crosshatetn final-test --confirm # une seule fois, quand tous les choix sont figés
    python -m crosshatetn demo --port 8000     # démonstration de bout en bout dans le navigateur
    python -m crosshatetn export-web           # prépare web/ (démo en ligne, Vercel)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import baseline
from .config import load_config
from .pipeline import prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crosshatetn", description="Socle de données et baselines CrossHateTN")
    parser.add_argument("--config", default="configs/data.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="sources, nettoyage, splits gelés, audit")
    p.add_argument("--refreeze", action="store_true", help="recrée les splits (à n'utiliser qu'en connaissance de cause)")
    p.add_argument("--no-strict", action="store_true", help="ne pas s'arrêter si l'audit échoue (déconseillé)")
    sub.add_parser("baseline", help="baselines E1 évaluées sur la validation")
    t = sub.add_parser("final-test", help="évaluation unique sur le test gelé")
    t.add_argument("--confirm", action="store_true")
    d = sub.add_parser("demo", help="démonstration de bout en bout dans le navigateur (127.0.0.1)")
    d.add_argument("--port", type=int, default=8000)
    w = sub.add_parser("export-web", help="prépare le dossier web/ pour la démo en ligne (Vercel)")
    w.add_argument("--dossier", default="web")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.command == "prepare":
        prepared = prepare(cfg, refreeze=args.refreeze, strict=not args.no_strict)
        print(prepared.summary().to_string())
        print(prepared.split_table().to_string())
        print(prepared.audit.table().to_string(index=False))
        return 0 if prepared.audit.ok else 1
    if args.command == "demo":
        from .demo import build_model, serve
        model = build_model(cfg)
        print(f"Modèle entraîné sur {model.n_train} commentaires ; Arabizi translittéré par règles.")
        serve(model, args.port)
        return 0
    if args.command == "export-web":
        from .demo import build_model
        from .web_export import export
        out = export(build_model(cfg), Path(args.dossier))
        print(f"Démo web prête dans {Path(args.dossier).resolve()} ; modèle : {out.name} "
              f"({out.stat().st_size // 1024} Ko) ; scores identiques à la démo locale.")
        return 0

    prepared = prepare(cfg, write=False, verbose=False)
    out_dir = cfg.path("outputs") / "baseline"
    if args.command == "baseline":
        results = baseline.run(prepared.labeled, cfg, out_dir)
        print(results.best.to_string(index=False))
        print(results.comparisons.to_string(index=False))
        print(results.report)
        return 0
    if not args.confirm:
        print("Refusé : ajouter --confirm, une seule fois, quand tous les choix sont figés.")
        return 2
    reference = json.loads((out_dir / "reference_e1.json").read_text(encoding="utf-8"))
    print(json.dumps(baseline.final_test(prepared.labeled, cfg, reference, confirm=True), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
