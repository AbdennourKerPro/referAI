# Pipeline SoccerNet Game State

## 1. Objectif

Cette phase ajoute une compréhension sémantique des personnes suivies sans fragiliser le
tracking. Le détecteur SportsMOT continue de détecter et suivre les personnes. Un second
modèle, entraîné sur des crops SoccerNet Game State, attribue ensuite un rôle à chaque
trajectoire :

- `player` ;
- `goalkeeper` ;
- `referee`.

La classe `other` est explicitement exclue de cette itération. Elle était trop hétérogène
et déséquilibrait fortement l'apprentissage. Un cas hors périmètre devra donc être
traité comme une prédiction peu confiante, et non comme une quatrième classe, jusqu'à la
création d'un vrai jeu de données `unknown` cohérent.

Les probabilités image par image sont lissées par `track_id` avec une moyenne mobile
exponentielle. Le rôle final est donc une propriété temporelle de l'agent, et non une
décision indépendante susceptible de changer à chaque image.

Le ballon reste une branche séparée. Ses annotations sont conservées dans
`ball_annotations.jsonl`, mais elles ne sont pas mélangées avec les crops humains du
classifieur de rôles.

## 2. Architecture livrée

```text
SoccerNet Game State v1.3
  -> téléchargement officiel et extraction sécurisée
  -> lecture de Labels-GameState.json
  -> validation des splits et de la version
  -> crops des personnes avec 30 % de contexte autour de bbox_image
  -> dataset de classification train/val/test
  -> plafonnement a 40 crops par trajectoire
  -> echantillonnage d'entrainement inversement pondere par frequence de role
  -> fine-tuning YOLO11s-cls selectionne sur macro-F1
  -> prédictions par crop
  -> lissage temporel par track_id
  -> moyenne des probabilites lissees par trajectoire
  -> métriques image, image lissée et trajectoire
```

Les informations suivantes restent dans les manifestes pour les phases ultérieures :

```text
sequence, image_id, frame_index, track_id, role, team, jersey,
bbox_image, bbox_pitch, source_image, crop_path
```

Les splits officiels sont conservés : `train` reste `train`, `valid` devient `val`, et
`test` reste `test`. Aucun crop d'une séquence de validation n'est placé dans
l'entraînement.

## 3. Fichiers ajoutés

| Fichier | Rôle |
|---|---|
| `scripts/download_soccernet_gamestate.py` | script autonome de téléchargement |
| `src/referai/soccernet.py` | téléchargement, extraction, validation et préparation |
| `src/referai/role_classification.py` | entraînement, validation et lissage temporel |
| `configs/train_soccernet_roles.yaml` | configuration du classifieur YOLO11s-cls |
| `requirements_soccernet.txt` | SDK officiel SoccerNet |
| `tests/test_soccernet_roles.py` | tests synthétiques du pipeline |

## 4. Pré-requis

Les vidéos et images SoccerNet sont soumises aux conditions indiquées sur la page
officielle. Remplir le NDA avant le téléchargement si SoccerNet le demande.

Sur le serveur RTX 3090 Ti :

```bash
cd /workspace/referAI
source /workspace/envs/refer/bin/activate
pip install -r requirements_soccernet.txt
pip install -e .
```

Le SDK officiel stable utilisé ici est `SoccerNet==0.1.62`. Les poids
`yolo11s-cls.pt` sont téléchargés automatiquement par Ultralytics au premier
entraînement si le fichier n'est pas déjà présent.

## 5. Télécharger SoccerNet Game State

La version documentée par le dépôt officiel est `gamestate-2024`, avec des annotations
Game State v1.3 :

```bash
referai-football download-soccernet \
  --output data/SoccerNetGS \
  --task gamestate-2024 \
  --split train \
  --split valid \
  --split test
```

Commande équivalente avec le script séparé :

```bash
python scripts/download_soccernet_gamestate.py \
  --output data/SoccerNetGS \
  --task gamestate-2024 \
  --split train \
  --split valid \
  --split test
```

Si un mot de passe NDA est demandé, ne pas l'écrire dans un fichier versionné :

```bash
export SOCCERNET_PASSWORD='MOT_DE_PASSE_RECU'
```

Le downloader lit cette variable sans l'afficher. Les archives sont conservées par
défaut. Ajouter `--delete-archives` uniquement si leur suppression après extraction est
souhaitée.

Si le SDK SoccerNet de l'environnement propose une édition plus récente, la tâche peut
être remplacée explicitement, par exemple `--task gamestate-2025`. Le préparateur reste
compatible tant que le fichier suit le schéma `Labels-GameState.json` et annonce une
version au moins égale à 1.3.

Arborescence attendue après extraction :

```text
data/SoccerNetGS/
  train/SNGS-*/
    img1/*.jpg
    Labels-GameState.json
  valid/SNGS-*/
    img1/*.jpg
    Labels-GameState.json
  test/SNGS-*/
```

## 6. Préparer les crops de rôles

Faire d'abord un test sur deux clips :

```bash
referai-football prepare-gamestate-roles \
  --source data/SoccerNetGS \
  --output data/processed/soccernet_roles_debug \
  --split train \
  --split valid \
  --frame-stride 10 \
  --max-samples-per-track 20 \
  --max-clips 2
```

Puis générer le dataset complet :

```bash
referai-football prepare-gamestate-roles \
  --source data/SoccerNetGS \
  --output data/processed/soccernet_roles_v2 \
  --split train \
  --split valid \
  --split test \
  --frame-stride 5 \
  --max-samples-per-track 40 \
  --context 0.30 \
  --minimum-version 1.3
```

`--frame-stride 5` conserve un crop toutes les cinq observations d'une trajectoire.
`--max-samples-per-track 40` évite qu'un long track de joueur domine excessivement le
dataset. `--context 0.30` conserve davantage de terrain autour de la personne,
information utile pour distinguer gardiens et arbitres. Pendant l'entraînement, un
sampler pondéré tire autant de crops attendus de chaque rôle ; les fichiers ne sont pas
dupliqués sur disque.

Sorties :

```text
data/processed/soccernet_roles_v2/
  train/{player,goalkeeper,referee}/*.jpg
  val/{player,goalkeeper,referee}/*.jpg
  test/{player,goalkeeper,referee}/*.jpg
  manifest.jsonl
  annotations.jsonl
  ball_annotations.jsonl
  dataset.yaml
  preparation_stats.json
```

Vérifier `preparation_stats.json` avant l'entraînement. En particulier, chaque rôle doit
avoir des crops dans `train` et `val`. Une classe vide signifie généralement que les
archives ou annotations ne correspondent pas au dataset attendu.

## 7. Entraîner le classifieur

```bash
referai-football train-role \
  --config configs/train_soccernet_roles.yaml \
  --hardware configs/hardware_3090Ti.yaml
```

Le modèle reste `yolo11s-cls.pt`, avec des entrées 224x224. Le meilleur checkpoint est
choisi avec la **macro-F1 de validation**, plutôt qu'avec l'accuracy : les gardiens et
arbitres ont ainsi le même poids que les joueurs dans `best.pt`. La patience courte, le
dropout et les augmentations de la configuration limitent le surapprentissage sans
réduire la capacité du modèle.

Le meilleur checkpoint est écrit dans :

```text
runs/classify/soccernet_roles_v2/weights/best.pt
```

Le tracking reste séquentiel, mais l'entraînement du classifieur peut utiliser le GPU et
des batches de crops.

Pour reprendre depuis `last.pt` :

```bash
referai-football train-role \
  --config configs/train_soccernet_roles.yaml \
  --hardware configs/hardware_3090Ti.yaml \
  --resume
```

Le chemin `runs/classify/soccernet_roles_v2/weights/last.pt` est déduit automatiquement.
La clé `resume_from` du YAML permet de fournir un autre checkpoint.

## 8. Évaluer image par image

```bash
referai-football validate-role \
  --weights runs/classify/soccernet_roles_v2/weights/best.pt \
  --data data/processed/soccernet_roles_v2 \
  --hardware configs/hardware_3090Ti.yaml \
  --split val \
  --imgsz 224 \
  --batch 64
```

Cette commande mesure le classifieur Ultralytics sans utiliser l'identité temporelle.
Elle affiche encore les accuracies top-1/top-5 pour information, mais expose aussi
`metrics/macro_f1` et utilise cette dernière comme `fitness`.

## 9. Évaluer le lissage temporel, la qualité et les priors spatiaux

```bash
referai-football predict-role-tracks \
  --weights runs/classify/soccernet_roles_v2/weights/best.pt \
  --data data/processed/soccernet_roles_v2 \
  --hardware configs/hardware_3090Ti.yaml \
  --output artifacts/role_eval/val \
  --split val \
  --alpha 0.20 \
  --batch 64 \
  --image-prior-strength 0.25 \
  --pitch-prior-strength 0.75
```

Si `dataset.yaml` référence une ancienne racine SoccerNet, ajouter par exemple
`--source-root data/SoccerNetGS`. Les datasets régénérés par
`prepare-gamestate-roles` enregistrent directement `image_width` et `image_height` dans
le manifeste et ne dépendent donc plus de ce chemin pour le prior image.

Sorties :

- `role_predictions.jsonl` : probabilités, qualité, positions, priors et contribution de
  chaque crop aux agrégations ;
- `track_role_predictions.jsonl` : quatre décisions et distributions par track ;
- `role_metrics.json` : accuracy, macro-F1, métriques par classe et matrices de confusion ;
- sections `track_baseline`, `track_quality_weighted`, `track_image_prior` et
  `track_pitch_oracle` pour comparer les quatre variantes.

Avec `alpha=0.20`, une nouvelle observation contribue à 20 % du nouvel état du track.
Une valeur plus faible stabilise davantage le rôle mais corrige plus lentement une
mauvaise initialisation. Le rôle `track_final` n'est plus la dernière prédiction du
track : il provient de la moyenne de toutes ses probabilités déjà lissées, ce qui évite
qu'une seule image tardive impose son rôle. `track_final` et les champs historiques
`aggregated_role` restent des alias du baseline non pondéré afin de ne pas modifier
silencieusement les résultats existants.

### 9.1 Qualité et contribution des observations

La qualité `fixed_v1` combine netteté, contraste, exposition, résolution du crop, taille
relative de la boîte, entropie, marge top-1/top-2 et confiance. L'apparence reçoit la
majorité du poids afin qu'une erreur très confiante mais floue ne domine pas un track.

- `quality.weight` contient le poids brut, borné par `--quality-minimum-weight` ;
- `aggregation_contributions` contient le poids normalisé dans le track ;
- `smoothed_probabilities` contient la distribution visuelle avant prior ;
- `image_fused_probabilities` et `pitch_oracle_fused_probabilities` contiennent les
  distributions après fusion.

### 9.2 Prior image utilisable sans calibration

Le point bas central de la boîte est normalisé par la largeur et la hauteur de l'image.
Un histogramme spatial par rôle est appris uniquement sur `train`, avec prior de classe
uniforme et lissage de Laplace. Ce prior est volontairement faible : la position dans
l'image dépend du cadrage, du zoom et des mouvements de caméra.

Les réglages sont `--image-prior-strength`, `--spatial-bins-x`,
`--spatial-bins-y` et `--spatial-smoothing`.

### 9.3 Prior terrain SoccerNet, oracle non déployable

Le prior `pitch_oracle` exploite `bbox_pitch` annoté par SoccerNet, notamment la distance
à la ligne de but. Il mesure la borne haute que pourrait apporter une bonne calibration.
Il ne constitue pas une fonctionnalité produit : une vidéo réelle ne fournit pas
`bbox_pitch`. Son usage restera interdit en production jusqu'à l'étape de calibration
image-terrain.

### 9.4 Protocole de sélection

1. Exécuter les quatre variantes sur `val`.
2. Comparer d'abord macro-F1 et rappel `goalkeeper` par track.
3. Régler les forces des priors uniquement sur `val`.
4. Figer `alpha`, les forces et les bins.
5. Exécuter une seule fois le réglage figé sur `test`.

`track_pitch_oracle` doit être rapporté séparément comme oracle et jamais comme la
performance déployable du système. Les critères principaux sont la macro-F1, le rappel
des gardiens et arbitres, puis le gain de chaque variante par rapport à
`track_baseline`.

## 10. Limites et suite

Cette livraison entraîne le rôle humain et conserve les autres attributs, mais ne les
prédit pas encore :

- `team` servira à un classifieur d'équipe séparé ;
- `jersey` servira à la reconnaissance de numéro ;
- `bbox_pitch` alimentera la calibration et les coordonnées métriques ;
- le ballon nécessitera un détecteur de petite cible séparé ;
- l'intégration temps réel du classifieur dans `track` viendra après validation de sa
  macro-F1 sur SoccerNet.

Ce découpage est volontaire : une classe visuelle instable ne doit pas casser les
identités ByteTrack. Le rôle est attaché au track après détection, puis lissé dans le
temps.
