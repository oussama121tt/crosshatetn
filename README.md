# CrossHateTN — socle de données et baseline

Projet P9 (module NLP, ESPRIM 5e année) : détection des discours haineux en tunisien, en alphabet arabe et en Arabizi.
Ce dépôt contient le **socle de données** (lot A) et la **baseline E1**, construits pour être reproductibles et sans fuite entre splits.

## Ce que fait le socle

- **Sources figées** : chaque corpus est récupéré à un commit précis et vérifié par SHA-256 ; un fichier modifié arrête tout.
- **Schéma commun** : `id` stable, `source`, `dialect`, `script`, `text_raw` (jamais modifié), `label_source`, `label` harmonisé (`normal`, `abusive`, `hate`).
- **Normalisation réversible** : `text_light` et `text_strong` à côté du texte brut ; la normalisation reste une variable d'expérience (E3).
- **Doublons** repérés sur une clé normalisée ; **conflits d'étiquettes** exclus jusqu'à arbitrage humain (`annotations/arbitrage.csv`).
- **Quasi-doublons** regroupés (similarité de caractères ≥ 0,70, toutes sources confondues) : un groupe ne traverse jamais deux splits.
- **Splits gelés** dans `splits/affectation.csv` (identifiants seulement, aucun texte) : stratifiés par source × classe, identiques pour toute l'équipe.
- **Audit des fuites** à chaque exécution : texte identique, clé normalisée, groupe ou paire proche partagés entre splits = arrêt.
- **Baseline E1** évaluée sur la validation tunisienne avec intervalles bootstrap ; le **test** n'est évalué que sur demande explicite, et chaque utilisation est journalisée.

## Démarrage rapide

### En local

```bash
pip install -r requirements.txt
python -m pytest                       # 45 tests, quelques secondes
python -m crosshatetn prepare          # socle de données (~30 s, Internet nécessaire)
python -m crosshatetn baseline         # baselines E1 sur la validation (~2 min)
```

Sous Windows, remplacer `python` par `py` si besoin. Lancer les commandes depuis la racine du dépôt.

### Démonstration

```bash
python -m crosshatetn demo --port 8000                                  # entraîne la référence (~15 s), puis ouvre http://127.0.0.1:8000
```

La page affiche chaque étape : écriture détectée, normalisation, n-grammes qui pèsent le plus, score du SVM, classe et latence. Les cas hors cadre (entrée vide, texte trop long, sans lettre, français, alphabet non pris en charge) sont refusés avant tout calcul. Une phrase en Arabizi est d'abord translittérée par les règles de `translit.py`. La démo tourne en local, sans LLM ni API.

### Démo en ligne (Vercel)

Le dossier `web/` contient la même démo, prête pour Vercel : une page (`index.html`) et une fonction Python (`api/classify.py`) qui n'utilise que numpy.

```bash
python -m crosshatetn export-web       # entraîne le modèle et écrit web/model/modele_demo.npz (~450 Ko)
```

La commande vérifie que la version en ligne donne exactement les mêmes scores que la démo locale. Sinon, elle s'arrête.

Tout automatiser sous Windows : double-clic sur `deploy.bat`. Le script crée le dépôt GitHub privé, pousse le code et déploie `web/` sur Vercel. À la première exécution, il ouvre la connexion GitHub puis Vercel dans le navigateur.

À la main, sur Vercel : *Add New → Project*, choisir le dépôt GitHub, puis **Root Directory = `web`**. Framework : *Other*. Aucune commande de build.

Le fichier du modèle ne contient aucun texte des corpus. Il contient en revanche des morceaux de 2 à 5 lettres et des poids appris sur T-HSAB, donc une donnée dérivée. Comme aucune licence n'est précisée, garder le dépôt GitHub **privé**. Vercel déploie aussi depuis un dépôt privé.

### Sur Kaggle

1. Mettre le dépôt sur GitHub (dépôt public possible : il ne contient aucun texte des corpus), **ou** ajouter le zip du dépôt comme dataset Kaggle (*Add Input → Upload*).
2. Importer `notebooks/kaggle_crosshatetn.ipynb` (*File → Import Notebook*) et, si GitHub, renseigner `REPO_URL` dans la première cellule.
3. Activer Internet dans les options de session, puis *Run All* (CPU suffit, ~3 min).

### Notebook Bronze (protocole complet)

`notebooks/bronze_crosshatetn.ipynb` refait tout le protocole Bronze depuis les sources. Il écrit `experiments.md`, les tableaux 1 à 3, les 30 erreurs catégorisées et le journal du test.

Entrées sur Kaggle :
- le zip du dépôt ;
- le fichier TUNIZI annoté ;
- les fichiers d'annotation remplis : `annotation_kappa_tunizi.xlsx` et `annotations_equipe.xlsx`.

Activer Internet. Le test n'est évalué que si `EVALUER_TEST = True`, et chaque évaluation est inscrite dans `splits/journal_test.csv`.

## Structure

```
configs/data.yaml            tous les paramètres du protocole (sources, seuils, splits, baseline)
src/crosshatetn/
  sources.py                 récupération figée + lecture des formats d'origine + schéma commun
  text.py                    normalisations, clé de doublon, détection d'écriture et indices Arabizi
  dedup.py                   doublons, conflits, arbitrage, groupes de quasi-doublons
  splits.py                  découpage stratifié par groupes, gel, manifeste
  audit.py                   contrôles de fuite (bloquants ou à revoir)
  metrics.py                 macro-F1, IC bootstrap, bootstrap apparié
  baseline.py                E1 : majoritaire, TF-IDF + régression logistique, TF-IDF + SVM
  lexique.py, translit.py    règles Bronze : lexique relu à la main, translittération Arabizi -> arabe
  demo.py                    démonstration locale dans le navigateur
  web_export.py, web_modele.py   préparation et inférence numpy de la démo en ligne
  pipeline.py, report.py     orchestration et rapport de données
annotations/arbitrage.csv    décisions humaines sur les conflits (versionné)
splits/                      affectation gelée, manifeste, journal du test (versionnés)
tests/                       tests unitaires (sans réseau)
notebooks/                   notebook Kaggle
web/                         démo en ligne (Vercel), générée par export-web
docs/                        protocole et carte de données
outputs/                     sorties régénérables (non versionnées)
```

## Sources

| Source | Rôle | Contenu | Classes | Licence |
|---|---|---|---|---|
| [T-HSAB](https://github.com/Hala-Mulki/T-HSAB-A-Tunisian-Hate-Speech-and-Abusive-Dataset) | référence tunisienne | 6 024 commentaires (alphabet arabe) | normal / abusive / hate | non spécifiée |
| [L-HSAB](https://github.com/Hala-Mulki/L-HSAB-First-Arabic-Levantine-HateSpeech-Dataset) | transfert multidialectal | 5 846 tweets levantins | normal / abusive / hate | non spécifiée |
| [TUNIZI](https://github.com/chaymafourati/TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset) | réserve Arabizi à annoter | 3 000 phrases | sentiment (pas haine) | non spécifiée |

Aucune licence n'étant précisée, les textes ne sont **jamais** versionnés ni republiés : ils sont retéléchargés à chaque exécution. Détails dans [docs/carte_donnees.md](docs/carte_donnees.md).

## Premiers résultats (validation T-HSAB, 894 messages)

| Modèle | Entraînement | macro-F1 [IC 95 %] | F1 `hate` |
|---|---|---|---|
| Majoritaire | tunisien | 0,259 [0,250 ; 0,267] | 0,000 |
| TF-IDF + SVM (**référence E1**) | tunisien | 0,678 [0,641 ; 0,716] | 0,520 |
| TF-IDF + régression logistique | tunisien | 0,700 [0,662 ; 0,735] | 0,564 |
| TF-IDF + SVM | tunisien + levantin | 0,706 [0,668 ; 0,742] | 0,577 |
| TF-IDF + régression logistique | tunisien + levantin | 0,718 [0,684 ; 0,754] | 0,597 |

- **Aperçu E2** : ajouter L-HSAB à l'entraînement améliore le SVM de +0,029 de macro-F1 (IC 95 % [0,007 ; 0,050], bootstrap apparié, p = 0,012).
- **Aperçu E3** : la normalisation n'apporte rien de mesurable (léger : −0,004, p = 0,58 ; fort : −0,010, p = 0,32).
- Ces chiffres servent à choisir, pas à conclure : `C` et la variante de texte ont été choisis sur cette même validation. Le chiffre à rapporter sera celui du test, évalué une seule fois.

## Règles de travail en équipe

- Ne jamais modifier `splits/` à la main : le manifeste le détecte et le pipeline s'arrête.
- Un conflit d'étiquettes se règle dans `annotations/arbitrage.csv`, puis on relance `prepare` et on versionne `annotations/` et `splits/`.
- Aucun choix (modèle, seuil, prétraitement) ne se fait sur le test. `final-test` exige `--confirm` et écrit dans `splits/journal_test.csv`.
- `python -m crosshatetn prepare --refreeze` recrée les splits : à réserver à une décision d'équipe, notée dans le rapport.

## Prochaines étapes (palier Bronze)

- Guide d'annotation et annotation pilote Arabizi sur la réserve TUNIZI (double annotation d'au moins 20 %, accord inter-annotateurs, arbitrage).
- Règles lexicales comme deuxième baseline.
- Catégorisation d'au moins 30 erreurs (`outputs/baseline/erreurs_validation_a_categoriser.csv`).
- Démonstration de bout en bout : en place (`python -m crosshatetn demo`), Arabizi translittéré par règles.

Voir [docs/protocole.md](docs/protocole.md) pour le détail des choix.
