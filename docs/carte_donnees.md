# Carte de données

Les effectifs ci-dessous viennent de l'exécution de référence du socle ; ils sont régénérés à chaque exécution dans `outputs/rapports/rapport_donnees.md`.

## Usage et précautions

- **Usage prévu** : recherche et enseignement, aide à la modération avec revue humaine. Jamais de sanction automatique.
- **Contenu sensible** : les corpus contiennent de vrais propos haineux et injurieux, visant notamment des groupes religieux, ethniques, des femmes et des minorités sexuelles. Accès encadré, avertissement avant consultation.
- **Licences** : aucune des trois sources n'a de fichier de licence. Usage interne au projet, citation obligatoire, aucune republication des textes ni de données dérivées sans l'accord des auteurs. Le dépôt ne contient que des identifiants.
- **Données personnelles** : les textes peuvent contenir des noms de personnes (personnalités publiques et particuliers). Ne pas les extraire ni les diffuser ; anonymiser tout exemple cité dans le rapport.

## T-HSAB (référence tunisienne)

| | |
|---|---|
| Dépôt | https://github.com/Hala-Mulki/T-HSAB-A-Tunisian-Hate-Speech-and-Abusive-Dataset |
| Version figée | commit `307f96f123980f1a308df8435979d82e5c4a8f57`, fichier `T-HSAB Corpus/T-HSAB.xlsx` |
| SHA-256 | `04b9503a1be25b2b9d137b4cbb74aa087abff74ed2ca247e9e384265219b4b84` |
| Citation | Haddad, Mulki et Oueslati (2019), *T-HSAB: A Tunisian Hate Speech and Abusive Dataset*, ICALP 2019, CCIS 1108, Springer |
| Collecte (selon le dépôt) | commentaires Facebook et YouTube, requêtes sur des cibles fréquentes de la haine (juifs, Africains, égalité dans l'héritage…) |
| Annotation (selon le dépôt) | 3 annotateurs tunisophones ; `normal`, `abusive`, `hate` (haine = langage injurieux + cible + dénigrement fondé sur l'identité) |
| Lignes | 6 024 (normal 3 820, abusive 1 126, hate 1 078) |
| Socle | 69 doublons, 4 conflits exclus, 5 951 conservées ; 100 % en alphabet arabe |
| Splits | train 4 164, validation 894, test 893 |

## L-HSAB (source levantine, transfert multidialectal)

| | |
|---|---|
| Dépôt | https://github.com/Hala-Mulki/L-HSAB-First-Arabic-Levantine-HateSpeech-Dataset |
| Version figée | commit `9abf6b6d05f484b67cd85a4a61476c51b5cc1910`, fichier `Dataset/L-HSAB` (TSV) |
| SHA-256 | `3fc5e06ab624b47e404a0530388631c4894c323ca038e726ce6dd3d0e6a371e3` |
| Citation | Mulki, Haddad, Bechikh Ali et Alshabani (2019), *L-HSAB: A Levantine Twitter Dataset for Hate Speech and Abusive Language*, 3rd Workshop on Abusive Language Online (ACL) |
| Collecte (selon le dépôt) | tweets politiques syriens et libanais (mars 2018 – février 2019) |
| Annotation (selon le dépôt) | 3 annotateurs levantins ; mêmes classes et mêmes définitions que T-HSAB |
| Lignes | 5 846 (normal 3 650, abusive 1 728, hate 468) |
| Socle | 114 doublons, 5 conflits exclus, 5 727 conservées ; 100 % en alphabet arabe |
| Splits | train 4 010, validation 859, test 858 |

Attention au décalage de domaine : sujets politiques levantins, Twitter, et beaucoup moins de `hate` (8 %) que dans T-HSAB (18 %).

## TUNIZI (réserve Arabizi à annoter)

| | |
|---|---|
| Dépôt | https://github.com/chaymafourati/TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset |
| Version figée | commit `33a341ce27d7db58198e40469f7cdc229ab3db89`, fichier `TUNIZI-Dataset.txt` |
| SHA-256 | `af5a8eab4f2e2e20d5ff86a1e79ac0903f7023c06d321b33660f8c371e1bd051` |
| Citation | Fourati, Messaoudi et Haddad (2020), *TUNIZI: a Tunisian Arabizi sentiment analysis Dataset*, AfricaNLP Workshop (ICLR 2020) |
| Collecte (selon le dépôt) | commentaires YouTube, prétraités par les auteurs (liens, émojis et ponctuation retirés) |
| Étiquettes d'origine | sentiment positif (1) / négatif (−1) : **pas** des étiquettes de haine |
| Lignes | 3 000 (1 500 / 1 500) |
| Socle | 76 doublons, 2 924 conservées ; 100 % en alphabet latin ; 1 813 avec chiffres-lettres ; 66 plutôt en français |

Les émojis et la ponctuation ayant été retirés à la source, ces textes ne ressemblent pas tout à fait à ceux de T-HSAB : à garder en tête pour E3 et E5.

## Sources écartées pour l'instant

- **ADHAR** (Frontiers in AI, 2024 ; 4 240 tweets, haine oui/non, plusieurs dialectes dont maghrébin) : l'article ne donne ni lien de téléchargement ni licence. Accès à demander aux auteurs.
- **OSACT 2020** (offensant / haine, arabe) : non intégrée à ce stade. Ses étiquettes sont binaires (offensant, haineux) et demanderaient une table de correspondance documentée.
