# referAI — détection et tracking football

Cette implémentation transforme le module décrit dans
`module_detection_tracking_football.md` en pipeline exécutable : préparation MOT,
fine-tuning YOLO11, ByteTrack, export JSON/JSONL/Parquet/MOT, vidéo annotée et
statistiques opérationnelles.

## Compatibilité GPU

Deux environnements sont volontairement figés :

| Machine | Environnement | Réglage initial | Précision |
|---|---|---|---|
| 3× Tesla K80, 11 Go par GPU, pilote R470/CUDA 11.4 | `requirements_K80.txt` | YOLO11m, 960 px, batch global 3 | FP32 |
| RTX 3090 Ti, 24 Go | `requirements_3090Ti.txt` (PyTorch cu118) | YOLO11m, 1280 px, batch 8 | AMP/FP16 |

Les mémoires des K80 ne s'additionnent pas : DDP réplique le modèle sur chaque
carte. Le batch K80 est donc `1 × 3 GPU`, et non un batch dimensionné pour 33 Go.
Le code détecte les GPU, limite le profil K80 à trois cartes, impose un batch global
divisible par le nombre de GPU et diminue batch puis résolution après une OOM.
L'AMP est désactivée sur Kepler : elle n'apporte pas les Tensor Cores d'Ampere et
évite plusieurs chemins FP16 anciens de cuDNN.

Le pilote R470 est la dernière branche pour Kepler et accepte CUDA 11.x. Le profil
K80 emploie les roues officielles PyTorch 1.12.1/cu113, compatibles avec le pilote
CUDA 11.4. Utiliser Python 3.9 pour K80 et Python 3.10 pour la 3090 Ti.

```bash
# K80
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements_K80.txt
pip install -e .

# ou 3090 Ti
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements_3090Ti.txt
pip install -e .
```

Vérifier le contexte avant un entraînement :

```bash
referai-football inspect-hardware --hardware configs/hardware_K80.yaml
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

Sur K80, la seconde commande doit notamment afficher `sm_37`. Sur 3090 Ti, elle
doit afficher `sm_86`.

## Préparer SoccerNet-Tracking ou SportsMOT

Le convertisseur attend le format MOT17 (`SEQUENCE/img1`, `gt/gt.txt`,
`seqinfo.ini`). Il crée une arborescence YOLO sans recopier les images par défaut.

```bash
referai-football prepare-mot \
  --source /data/SoccerNet/tracking \
  --output /data/processed/soccernet_yolo \
  --split-strategy by-match \
  --match-map match_map.csv \
  --class-map configs/class_map_mot.yaml
```

`match_map.csv` relie chaque clip à son match et empêche les fuites :

```csv
sequence,match_id
SNMOT-001,game_01
SNMOT-002,game_01
SNMOT-003,game_02
```

Avec SportsMOT, les splits officiels sont déjà des dossiers `train/val/test` :

```bash
referai-football prepare-mot \
  --source /data/SportsMOT/dataset \
  --output /data/processed/sportsmot_yolo \
  --sequence-list /data/SportsMOT/splits_txt/football.txt \
  --split-strategy existing
```

SoccerNet-Tracking et SportsMOT ne portent pas, dans leur GT MOT officielle, les
quatre classes détaillées du cahier des charges. Le mapping fourni les convertit donc
en `player`. Pour apprendre `goalkeeper`, `referee` et surtout `ball`, il faut ajouter
les annotations enrichies prévues dans la spécification et adapter
`configs/class_map_mot.yaml`.

Pour la phase ballon, les images contenant la classe 3 peuvent être répétées sans
dupliquer les fichiers :

```bash
# Générer d'abord les variantes vidéo réalistes (jeu d'entraînement uniquement)
referai-football augment \
  --data /data/processed/soccernet_yolo/data.yaml \
  --copies 1 --seed 42

# Puis sur-échantillonner le ballon, variantes comprises
referai-football oversample \
  --data /data/processed/soccernet_yolo/data.yaml \
  --class-id 3 --factor 4
```

Le résultat est `data_oversampled.yaml`. Le sur-échantillonnage agit comme une
pondération de classe compatible avec le trainer Ultralytics, sans modifier sa loss
interne.

La commande `augment` ne touche pas à la validation/test et conserve les boîtes :
elle ajoute luminosité/couleur, flou de mouvement, compression JPEG, réduction de
résolution et petites occultations. Les transformations géométriques cohérentes avec
les boîtes restent gérées par YOLO via les trois YAML : translation, zoom, changement
d'échelle, retournement et mosaïque décroissante. La mosaïque est fermée sur les
dernières epochs et n'est jamais appliquée au tracker.

## Entraîner les trois phases

Renseigner d'abord les chemins `data` dans les YAML. Pour chaque phase suivante,
remplacer `model` par le `best.pt` de la phase précédente.

```bash
# Phase 1 : adaptation SportsMOT
referai-football train --config configs/train_sportsmot.yaml \
  --hardware configs/hardware_K80.yaml

# Phase 2 : domaine SoccerNet
referai-football train --config configs/train_soccernet.yaml \
  --hardware configs/hardware_K80.yaml

# Phase 3 : ballon et cas difficiles
referai-football train --config configs/train_ball.yaml \
  --hardware configs/hardware_K80.yaml
```

Pour la 3090 Ti, remplacer uniquement le fichier matériel par
`configs/hardware_3090Ti.yaml`. Une exécution interrompue se reprend avec `--resume`
et un `model` pointant sur `last.pt`.

Les réglages lourds sont explicitement distincts : 3 processus DDP sur K80,
micro-batch 1 et FP32 ; un processus sur 3090 Ti, batch 8 et AMP. Si YOLO11m reste
trop lent sur K80, `yolo11s.pt` peut être choisi sans changer le pipeline.

## Valider et tracker une vidéo

```bash
referai-football validate \
  --weights runs/detect/phase3_ball/weights/best.pt \
  --data /data/processed/soccernet_yolo/data.yaml \
  --hardware configs/hardware_3090Ti.yaml --split test

referai-football track \
  --video match.mp4 \
  --weights runs/detect/phase3_ball/weights/best.pt \
  --tracker configs/bytetrack_football.yaml \
  --hardware configs/hardware_3090Ti.yaml \
  --output artifacts/match.jsonl \
  --annotated-video artifacts/match_annotated.mp4 \
  --mot-output artifacts/match_mot.txt \
  --trajectories-output artifacts/trajectories.json
```

Formats de sortie :

- `.json` : tableau de frames au schéma demandé ;
- `.jsonl` : même schéma, une frame par ligne, recommandé pour les longs matchs ;
- `.parquet` : une ligne par objet et par frame, compression Zstandard ;
- MOT : format 10 colonnes accepté par TrackEval/SoccerNet ;
- `*.stats.json` : FPS, latence moyenne et pic de mémoire GPU.

ByteTrack conserve 90 frames (environ 3,6 s à 25 fps), utilise les détections dès
0,05 et ne crée une piste qu'à 0,45. Ces seuils doivent être ajustés sur validation.
Le tracking d'une seule vidéo reste sur un GPU car son état est séquentiel ; les GPU
restants servent à DDP pendant l'entraînement ou à traiter plusieurs vidéos via des
processus séparés.

### HOTA, IDF1, MOTA et AssA

`--mot-output` produit exactement les 10 colonnes MOTChallenge. Le convertisseur
conserve également `mot_gt/<split>/<sequence>/gt/gt.txt`, `seqinfo.ini` et les
`seqmaps`. Ces fichiers sont destinés à l'implémentation officielle TrackEval :

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
cd TrackEval
python scripts/run_mot_challenge.py \
  --METRICS HOTA CLEAR Identity \
  --DO_PREPROC False \
  --GT_FOLDER /data/processed/soccernet_yolo/mot_gt \
  --TRACKERS_FOLDER /data/predictions
```

L'arborescence finale doit suivre le format de benchmark personnalisé documenté par
TrackEval. Les familles `HOTA`, `CLEAR` et `Identity` donnent notamment HOTA/AssA,
MOTA/fragmentations et IDF1/changements d'identité. Le prétraitement est désactivé
car les classes distractrices MOT17 ne correspondent pas au protocole SoccerNet.

## Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest
```

Les tests unitaires n'ont besoin ni de GPU ni des poids YOLO.
