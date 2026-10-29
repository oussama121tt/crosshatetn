# Protocole du socle de données et de l'évaluation

Tous les paramètres cités ici sont dans `configs/data.yaml`. Les chiffres à jour sont régénérés dans `outputs/rapports/rapport_donnees.md`.

## 1. Récupération des sources

- Chaque source est figée par son dépôt, un **commit** et l'**empreinte SHA-256** du fichier.
- Téléchargement direct (`raw.githubusercontent.com`) puis, en secours, `git fetch` du commit seul.
- Empreinte différente : le fichier est supprimé et le pipeline s'arrête. On ne travaille jamais sur une version non vérifiée.
- L'identifiant d'une ligne est `<source>-<rang>` (rang de l'enregistrement dans le fichier figé, à partir de 1). Il est donc stable tant que la version de la source ne change pas.

## 2. Lecture et harmonisation

| Source | Format | Particularités traitées |
|---|---|---|
| T-HSAB | Excel sans en-tête (texte, étiquette) | — |
| L-HSAB | TSV `Tweet`, `Class` | 3 tweets entre guillemets sur plusieurs lignes (lecture CSV avec guillemets) |
| TUNIZI | `<1\|-1>;<phrase>` par ligne | 2 lignes avec un caractère parasite avant l'étiquette (`%1;`, `"1;`), ignoré sans jamais avaler le signe moins |

Les étiquettes de T-HSAB et L-HSAB sont identiques (`normal`, `abusive`, `hate`, mêmes définitions, mêmes auteurs) : la table d'harmonisation est l'identité, mais elle reste explicite (`label_map`) et toute étiquette imprévue arrête le chargement. TUNIZI est étiquetée en sentiment : elle n'a pas d'étiquette harmonisée et sert de réserve à annoter.

## 3. Normalisation (réversible)

| Colonne | Opérations | Usage |
|---|---|---|
| `text_raw` | aucune | référence, entrée possible des modèles |
| `text_light` | NFKC, suppression des diacritiques et du tatweel, alefs unifiés (آ أ إ ٱ → ا), espaces | entrée possible des modèles, quasi-doublons |
| `text_strong` | léger + minuscules, ى → ي, ة → ه, lettres répétées 3+ fois réduites à 2 | entrée possible des modèles |
| `dedup_key` | fort + suppression de la ponctuation et des émojis, toute répétition réduite à 1 | détection des doublons uniquement |

Les injures, la ponctuation et les émojis restent dans les entrées des modèles : ce sont des indices.

Le système d'écriture (`script`) vaut `arabic`, `latin`, `mixed` (au moins 2 lettres de chaque alphabet) ou `other`. Il mesure l'alphabet, pas le dialecte. Pour les textes latins, deux indices aident à repérer l'Arabizi : `arabizi_digits` (chiffres-lettres comme 3, 7, 9 collés à une lettre) et `french_ratio` (part de mots outils français).

## 4. Doublons et conflits

- **Doublon** : même `dedup_key` dans une même source. On garde la ligne de plus petit rang.
- **Conflit** : un groupe de doublons avec des étiquettes différentes. Politique par défaut `exclude` : le texte est écarté de tous les splits tant qu'un humain ne l'a pas arbitré.
- **Arbitrage** : une ligne par conflit dans `annotations/arbitrage.csv` (`id_representant`, `label_arbitre`, `arbitre`, `date`, `commentaire`). Le fichier est versionné ; une étiquette inconnue ou un double arbitrage arrête le pipeline.
- La liste des conflits à arbitrer est régénérée dans `outputs/data/conflits_a_arbitrer.csv`, avec la majorité stricte quand elle existe.

## 5. Quasi-doublons et groupes

- Similarité cosinus TF-IDF sur n-grammes de caractères `char_wb` de 3 à 5, calculée sur `text_light`, **toutes sources confondues** (un tweet levantin proche d'un commentaire tunisien du test serait une fuite pour l'entraînement multidialectal).
- Paires ≥ 0,70 : reliées en groupes (composantes connexes). Le seuil a été choisi en inspectant des paires entre 0,70 et 0,85 : on y trouve des reprises du même message avec un mot en plus ou en moins, des ordres de mots différents et des fautes de frappe. Le plus grand groupe compte 13 textes, ce qui ne déséquilibre pas les splits.
- Paires dans [0,60 ; 0,70[ entre splits différents : non bloquantes, listées dans `outputs/data/quasi_doublons_a_revoir.csv` pour une revue humaine.
- Limite connue : ce contrôle ne voit pas les translittérations arabe ↔ Arabizi. Il faudra l'étendre (comparaison après translittération) quand des textes Arabizi annotés seront ajoutés.

## 6. Splits

- Textes étiquetés conservés de toutes les sources, découpés ensemble avec `StratifiedGroupKFold` (20 plis, graine 42) : strates = source × classe, groupes = quasi-doublons. 3 plis pour le test, 3 pour la validation, 14 pour l'entraînement (≈ 70 / 15 / 15 pour chaque source et chaque classe).
- L'affectation est **gelée** dans `splits/affectation.csv` (identifiants seulement) avec un manifeste : effectifs, empreintes des listes d'identifiants de chaque split, version de scikit-learn, empreinte de la configuration des données, historique.
- Aux exécutions suivantes, l'affectation est relue, jamais recalculée. Un nouvel identifiant (conflit arbitré, nouvelle source) prend le split de son groupe ; s'il est seul, il est placé par hachage déterministe selon les mêmes proportions. L'ajout est inscrit dans l'historique.
- Si la liste d'un split ne correspond plus au manifeste (modification à la main) ou si un groupe se retrouve à cheval sur deux splits, le pipeline s'arrête.

## 7. Audit des fuites (à chaque exécution)

Bloquants : identifiants uniques ; un split valide par texte ; aucun groupe partagé ; aucun texte brut identique entre splits ; aucune `dedup_key` identique entre splits ; aucune paire de similarité ≥ 0,70 entre splits.
Non bloquants (« à revoir ») : paires dans [0,60 ; 0,70[ ; écart de proportion d'une classe entre un split et sa source supérieur à 2 points ; taille des groupes.

## 8. Évaluation

- **Validation tunisienne** (T-HSAB) pour tous les choix : variante de texte, `C`, modèle, jeu d'entraînement.
- **Métrique principale** : macro-F1. On suit aussi le F1 et le rappel de `hate`, la précision et le rappel par classe, et la matrice de confusion.
- **Incertitude** : intervalle de confiance bootstrap à 95 % (1 000 rééchantillonnages, percentiles). Deux systèmes se comparent par **bootstrap apparié** sur les mêmes exemples (différence de macro-F1, IC 95 %, p-valeur bilatérale).
- **Graines** : les modèles linéaires de la baseline sont déterministes, donc trois graines donneraient trois fois le même résultat ; leur variabilité est estimée par bootstrap. La moyenne ± écart-type sur trois graines s'appliquera aux encodeurs fine-tunés (palier Silver).
- **Optimisme de sélection** : les scores de validation des configurations retenues sont légèrement optimistes, puisqu'elles ont été choisies sur cette validation.
- **Test gelé** : évalué uniquement par `python -m crosshatetn final-test --confirm` (ou `final_test(..., confirm=True)`), sur la configuration de référence figée. Chaque évaluation est ajoutée à `splits/journal_test.csv`, à versionner.

## 9. Baseline E1

- Représentation : TF-IDF sur mots (1–2) + TF-IDF sur caractères `char_wb` (2–5, `min_df=2`), ajustée sur l'entraînement seul.
- Modèles : majoritaire ; régression logistique et SVM linéaire avec `class_weight="balanced"` (la classe `hate` est minoritaire).
- Grilles : `C` ∈ {0,03 ; 0,1 ; 0,3 ; 1 ; 3} pour le SVM, {0,3 ; 1 ; 3 ; 10 ; 30} pour la régression logistique.
- Jeux d'entraînement : `tn` (T-HSAB) et `tn+lev` (T-HSAB + L-HSAB), toujours évalués sur la validation tunisienne.
- Référence E1 : le meilleur SVM entraîné sur `tn` ; ses erreurs de validation sont exportées pour l'audit des erreurs.
