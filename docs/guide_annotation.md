# Guide d'annotation CrossHateTN

Schéma de T-HSAB, à 3 étiquettes. Pour l'évaluation en 2 classes : **haineux = abusive + hate**, **non haineux = normal**.
Une seule étiquette par phrase, choisie après lecture de la phrase entière.

## Étiquettes

| Étiquette | Définition | Exemple |
|---|---|---|
| `hate` | Attaque, insulte, déshumanisation ou appel à exclure une personne ou un groupe **à cause de son identité** : religion, origine, nationalité, région, couleur, genre, orientation sexuelle, handicap. | «كافر لعدم تطبيقه القانون الجديد» |
| `abusive` | Insulte, grossièreté ou mépris **sans** viser une identité. | «يا بهيم سكر فمك» · *tfih 3al 97aab* |
| `normal` | Tout le reste, y compris une critique sans insulte, une citation ou une mention d'insulte, une négation, un sens littéral. | «قالي يا بهيم وضحكنا» |

## Règles de décision

1. Une cible et une identité visée : `hate`. Une insulte sans identité visée : `abusive`.
2. Insulte **citée ou rapportée**, sans être reprise à son compte : `normal`.
3. Insulte **niée** («إنتي موش بهيم») ou mot pris au **sens littéral** (l'animal) : `normal`.
4. Juron ou grossièreté sans cible : `abusive`.
5. Moquerie ou ironie qui rabaisse une personne, même sans gros mot : `abusive`.
6. Plaisanterie entre amis évidente : `normal`. En cas de doute : `abusive`, avec un commentaire.
7. Arabizi : lire la phrase à voix haute. 3 = ع, 7 = ح, 9 = ق, 5 = خ, 2 = ء, 8 = غ.
8. Ne jamais laisser une ligne vide : choisir l'étiquette la plus probable et écrire le doute en commentaire.

## Conditions de la double annotation (kappa)

- Le second annotateur n'est pas celui qui a fait la première annotation.
- Il ne voit pas les premières étiquettes et n'utilise aucun outil (traducteur, LLM).
- Les phrases sont tirées au hasard (graine 42) dans TUNIZI, sans choix manuel.
- Le kappa de Cohen est calculé en 3 étiquettes et en 2 classes, avec l'effectif annoté.
