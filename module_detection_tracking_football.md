# Module 1 — Détection et suivi des acteurs du jeu

## 1. Finalité du module

L’objectif est de construire un système de vision capable d’analyser une séquence vidéo de football et de produire, pour chaque image, la position et l’identité temporelle des principaux objets présents sur le terrain.

Ce module constituera la fondation des futures fonctions d’assistance à l’arbitrage :

- détection des contacts et des fautes ;
- analyse des hors-jeu ;
- suivi de la position du ballon ;
- identification du dernier défenseur ;
- détection des sorties de balle ;
- reconstruction des trajectoires ;
- analyse des comportements et événements de jeu.

À ce stade, le système ne prend aucune décision arbitrale. Il produit uniquement des observations visuelles structurées et traçables.

---

## 2. Entrée du système

### Entrée principale

Une vidéo de match de football issue d’une caméra télévisée principale.

Caractéristiques visées :

- format MP4 ou flux vidéo décodé ;
- résolution typique : `1920 × 1080` pixels ;
- fréquence : 25 à 30 images par seconde ;
- caméra mobile avec panoramiques, zooms et changements d’échelle ;
- présence possible d’occultations, de flou de mouvement et de joueurs de petite taille.

Pour l’entraînement, la vidéo est décomposée en images ou en courtes séquences ordonnées.

### Entrée d’une itération

Pour une image prise à l’instant `t` :

$$
I_t \in \mathbb{R}^{H \times W \times 3}
$$

avec :

- `H` : hauteur de l’image ;
- `W` : largeur de l’image ;
- `3` : canaux rouge, vert et bleu.

Le détecteur traite chaque image individuellement. Le tracker exploite ensuite les détections courantes ainsi que l’historique des images précédentes.

---

## 3. Sortie attendue

Pour chaque objet `i` détecté à l’image `t` :

$$
O_i^{(t)} = \left(x_1, y_1, x_2, y_2, c, s, \mathrm{id}\right)
$$

avec :

- `(x1, y1)` : coin supérieur gauche de la boîte ;
- `(x2, y2)` : coin inférieur droit ;
- `c` : classe de l’objet ;
- `s` : score de confiance ;
- `id` : identifiant temporel attribué par le tracker.

### Classes initiales

La première version doit reconnaître :

1. `player` — joueur de champ ;
2. `goalkeeper` — gardien ;
3. `referee` — arbitre ;
4. `ball` — ballon.

Une classe `staff` ou `other` pourra être ajoutée afin de réduire les faux positifs provenant du bord du terrain.

### Sortie structurée

```json
{
  "frame_id": 1842,
  "timestamp": 73.68,
  "objects": [
    {
      "track_id": 17,
      "class": "player",
      "confidence": 0.94,
      "bbox": [812, 327, 861, 451]
    },
    {
      "track_id": 4,
      "class": "referee",
      "confidence": 0.88,
      "bbox": [1035, 298, 1076, 426]
    },
    {
      "track_id": 201,
      "class": "ball",
      "confidence": 0.63,
      "bbox": [945, 382, 957, 394]
    }
  ]
}
```

### Sorties temporelles

Pour chaque objet suivi :

$$
T_i = \left\{\left(t, x_t, y_t, w_t, h_t\right)\;\middle|\; t_0 \le t \le t_1\right\}
$$

Cette trajectoire permettra ensuite de calculer :

- la vitesse apparente ;
- la direction du déplacement ;
- les périodes d’occultation ;
- les interactions entre joueurs ;
- la distance entre un joueur et le ballon.

---

## 4. Architecture proposée

Le système suit une architecture de type **tracking-by-detection**.

### Étape A — Détection

Un détecteur mono-image de la famille YOLO reçoit chaque image et prédit :

- les boîtes englobantes ;
- les classes ;
- les scores de confiance.

Modèle de départ recommandé :

- YOLO11-M ou modèle équivalent de taille moyenne ;
- initialisation avec des poids préentraînés sur COCO ;
- adaptation au domaine du football par fine-tuning.

Un modèle de taille moyenne représente un bon compromis entre précision, vitesse et consommation mémoire pour un GPU de 24 Go.

### Étape B — Suivi multi-objets

ByteTrack est utilisé comme première baseline de tracking.

À chaque nouvelle image, le tracker :

1. prédit la nouvelle position des trajectoires existantes ;
2. associe les détections à forte confiance aux trajectoires ;
3. utilise certaines détections à faible confiance pour récupérer les objets partiellement occultés ;
4. crée de nouvelles trajectoires ;
5. conserve temporairement les trajectoires disparues ;
6. attribue un identifiant stable à chaque objet.

### Extension prévue

Dans une seconde itération, ByteTrack pourra être remplacé ou complété par un tracker intégrant un réseau de ré-identification :

- BoT-SORT ;
- DeepSORT ;
- OC-SORT avec ReID ;
- tracker spécialisé développé pour le projet.

Cette extension utilisera conjointement :

- la position ;
- le mouvement ;
- l’apparence du joueur ;
- la couleur du maillot ;
- l’équipe probable.

---

## 5. Dataset principal

### SoccerNet-Tracking

Le dataset principal sera **SoccerNet-Tracking**, car il correspond directement au domaine étudié.

Il est adapté aux difficultés spécifiques du football :

- caméra mobile ;
- changements d’échelle ;
- mouvements rapides ;
- occultations ;
- suivi de longue durée ;
- plans larges de diffusion télévisée.

SoccerNet-Tracking sera utilisé pour :

- l’entraînement final du détecteur ;
- l’ajustement des paramètres du tracker ;
- l’évaluation du suivi dans des conditions réelles de football.

### Dataset complémentaire : SportsMOT

**SportsMOT** pourra être utilisé pour un préentraînement ou une phase d’augmentation des données consacrée au suivi des joueurs.

Il est particulièrement utile pour apprendre à gérer :

- les mouvements rapides ;
- les joueurs visuellement similaires ;
- les croisements ;
- les changements de caméra ;
- les fortes densités d’objets.

En revanche, il ne suffit pas à lui seul pour entraîner correctement les classes `ball`, `referee` et `goalkeeper`.

### Données supplémentaires

Une annotation complémentaire sera probablement nécessaire pour le ballon, car :

- il est très petit ;
- il peut occuper moins de quelques dizaines de pixels ;
- il est fréquemment occulté ;
- sa vitesse provoque du flou ;
- les datasets de tracking de joueurs ne garantissent pas toujours une couverture suffisante.

Un sous-ensemble de séquences pourra donc être réannoté ou vérifié manuellement pour la classe `ball`.

---

## 6. Stratégie d’entraînement du détecteur

### Initialisation

Le détecteur ne sera pas entraîné depuis zéro.

Il sera initialisé avec des poids préentraînés sur un dataset généraliste, puis affiné sur les images de football.

$$
\theta_{\mathrm{initial}} = \theta_{\mathrm{pretrained}}
$$

Cette stratégie réduit :

- le temps d’entraînement ;
- la quantité de données nécessaires ;
- le risque de surapprentissage.

### Phase 1 — Adaptation générale au sport

Fine-tuning sur les images de football de SportsMOT :

- apprentissage de la détection des joueurs ;
- adaptation aux plans larges ;
- adaptation aux petites personnes ;
- adaptation aux mouvements de caméra.

### Phase 2 — Adaptation spécifique au football

Fine-tuning sur SoccerNet-Tracking :

- amélioration sur les images de diffusion télévisée ;
- apprentissage des classes spécifiques ;
- ajustement aux tailles d’objets rencontrées dans les matchs.

### Phase 3 — Ballon et cas difficiles

Sur-échantillonnage des images contenant :

- un ballon visible ;
- des joueurs fortement occultés ;
- des regroupements ;
- des plans très larges ;
- du flou de mouvement ;
- des zones proches des lignes du terrain.

---

## 7. Fonction de coût du détecteur

La perte totale du détecteur est :

$$
\mathcal{L}_{\mathrm{det}}
=
\lambda_{\mathrm{box}}\mathcal{L}_{\mathrm{box}}
+
\lambda_{\mathrm{cls}}\mathcal{L}_{\mathrm{cls}}
+
\lambda_{\mathrm{dfl}}\mathcal{L}_{\mathrm{dfl}}
$$

avec :

- `L_box` : perte de localisation des boîtes ;
- `L_cls` : perte de classification ;
- `L_dfl` : Distribution Focal Loss ;
- `λ_box`, `λ_cls` et `λ_dfl` : coefficients de pondération.

### Perte de localisation

`L_box` mesure la différence entre la boîte prédite et la boîte réelle.

Une perte basée sur l’IoU, comme CIoU, peut prendre en compte :

- le recouvrement ;
- la distance entre les centres ;
- le rapport hauteur-largeur.

Elle est particulièrement importante pour :

- les joueurs proches les uns des autres ;
- les regroupements ;
- la localisation précise du ballon.

### Perte de classification

`L_cls` pénalise les erreurs sur les classes :

- `player` ;
- `goalkeeper` ;
- `referee` ;
- `ball`.

Une Binary Cross-Entropy ou une variante de Focal Loss peut être utilisée.

La Focal Loss devient pertinente lorsque le déséquilibre entre classes est important.

### Distribution Focal Loss

`L_dfl` modélise les coordonnées des boîtes comme des distributions discrètes plutôt que comme de simples valeurs régressées.

Elle améliore la précision de localisation des frontières des boîtes.

### Pondération des classes

Pour limiter l’effet du déséquilibre entre classes :

$$
\mathcal{L}_{\mathrm{cls}}
=
-\sum_c w_c\,y_c\,\log(p_c)
$$

avec :

- `c` : classe considérée ;
- `w_c` : poids associé à la classe ;
- `y_c` : valeur réelle ;
- `p_c` : probabilité prédite pour la classe.

Les classes rares comme `ball`, `goalkeeper` et `referee` peuvent recevoir un poids supérieur à celui de la classe `player`.

Le ballon ne doit cependant pas recevoir un poids excessif, au risque d’augmenter fortement les faux positifs.

---

## 8. Entraînement du tracker

Dans la première version, ByteTrack ne nécessite pas d’entraînement profond supplémentaire.

Il utilise :

- les boîtes du détecteur ;
- leurs scores ;
- une prédiction de mouvement ;
- une mesure d’association spatiale ;
- des seuils de confiance ;
- une durée de conservation des trajectoires perdues.

Les principaux hyperparamètres seront ajustés sur l’ensemble de validation :

- seuil de détection haute confiance ;
- seuil de détection basse confiance ;
- seuil d’IoU pour l’association ;
- durée maximale de conservation d’une trajectoire perdue ;
- seuil de création d’une nouvelle trajectoire.

Cela permet de séparer clairement :

1. la qualité de la détection ;
2. la qualité de l’association temporelle.

---

## 9. Extension avec ré-identification

ByteTrack risque de changer l’identité d’un joueur lors :

- d’une occultation prolongée ;
- d’un croisement ;
- d’une sortie puis d’un retour dans le champ ;
- d’un changement brutal de cadrage.

Une seconde version ajoutera un encodeur d’apparence.

$$
e_i = f_{\phi}\!\left(\mathrm{crop}_i\right)
$$

avec :

- `crop_i` : image recadrée autour du joueur ;
- `f_φ` : réseau de ré-identification ;
- `e_i` : vecteur d’embedding du joueur.

### Fonction de coût de ré-identification

$$
\mathcal{L}_{\mathrm{reid}}
=
\lambda_{\mathrm{id}}\mathcal{L}_{\mathrm{CE}}
+
\lambda_{\mathrm{tri}}\mathcal{L}_{\mathrm{triplet}}
$$

avec :

- `L_CE` : perte de classification d’identité ;
- `L_triplet` : Triplet Loss ;
- `λ_id` et `λ_tri` : coefficients de pondération.

### Triplet Loss

$$
\mathcal{L}_{\mathrm{triplet}}
=
\max\left(
0,\,
d(e_a,e_p)-d(e_a,e_n)+m
\right)
$$

avec :

- `e_a` : embedding de l’ancre ;
- `e_p` : embedding du même joueur ;
- `e_n` : embedding d’un autre joueur ;
- `d` : fonction de distance ;
- `m` : marge minimale.

L’objectif est :

$$
d(e_a,e_p) < d(e_a,e_n)
$$

Deux observations du même joueur doivent donc être plus proches dans l’espace d’embedding que deux observations de joueurs différents.

---

## 10. Augmentations de données

Les augmentations devront reproduire les conditions réelles de diffusion :

- variation de luminosité et de contraste ;
- variation de saturation ;
- balance des couleurs ;
- flou de mouvement ;
- bruit de compression ;
- réduction de résolution ;
- redimensionnement multi-échelle ;
- occultations artificielles ;
- recadrage ;
- translation ;
- zoom.

Les transformations géométriques devront rester cohérentes avec les boîtes annotées.

Les augmentations de type mosaïque devront être utilisées avec prudence. Elles peuvent améliorer la détection, mais produisent des scènes temporellement irréalistes et ne doivent pas être utilisées directement pour entraîner le tracker.

---

## 11. Séparation des données

La séparation doit être effectuée par match, et non par image.

Exemple :

- 70 % des matchs pour l’entraînement ;
- 15 % pour la validation ;
- 15 % pour le test.

Cette stratégie évite que des images presque identiques d’un même match apparaissent à la fois dans l’entraînement et le test.

Elle réduit également les fuites liées :

- aux mêmes joueurs ;
- aux mêmes maillots ;
- au même stade ;
- au même éclairage ;
- à la même réalisation télévisée.

---

## 12. Métriques d’évaluation

### Détection

Le détecteur sera évalué avec :

- précision ;
- rappel ;
- AP par classe ;
- mAP@0.50 ;
- mAP@0.50:0.95 ;
- rappel spécifique du ballon.

Le rappel est particulièrement important pour un assistant à l’arbitrage : un objet manqué peut empêcher toute analyse ultérieure.

### Tracking

Les principales métriques seront :

- HOTA ;
- IDF1 ;
- MOTA ;
- AssA ;
- nombre de changements d’identité ;
- fragmentation des trajectoires.

### Métriques opérationnelles

Le système devra également mesurer :

- nombre d’images traitées par seconde ;
- latence par image ;
- mémoire GPU utilisée ;
- stabilité sur des séquences longues ;
- taux de trajectoires interrompues ;
- rappel du ballon selon sa taille apparente.

---

## 13. Configuration d’entraînement initiale

Configuration indicative pour un GPU de 24 Go :

- résolution : `1280 × 720` ou entrée carrée de 1280 pixels ;
- batch size : 4 à 12 selon le modèle ;
- précision mixte : FP16 ou BF16 ;
- optimiseur : AdamW ou SGD ;
- nombre d’epochs : 50 à 150 ;
- scheduler : cosine decay ;
- early stopping sur la validation ;
- gradient accumulation si nécessaire ;
- sauvegarde du meilleur checkpoint.

Exemple de métrique composite :

$$
S
=
0{,}35\,AP_{\mathrm{players}}
+
0{,}20\,AP_{\mathrm{referee}}
+
0{,}15\,AP_{\mathrm{goalkeeper}}
+
0{,}30\,AP_{\mathrm{ball}}
$$

Le ballon reçoit une pondération importante, car sa détection sera indispensable aux futurs modules d’assistance à l’arbitrage.

---

## 14. Pipeline général

```text
Vidéo
  → Détecteur YOLO affiné
  → Détections par image
  → ByteTrack
  → Trajectoires structurées
  → Fichier JSON ou Parquet
  → Vidéo annotée
```

---

## 15. Modèle objectif de la première version

La première version complète sera :

```text
Vidéo
  → Détecteur supervisé
  → Tracker multi-objets
  → Trajectoires exploitables
```

Elle produira :

- les boîtes des joueurs, gardiens, arbitres et du ballon ;
- un identifiant stable pour chaque objet ;
- les trajectoires image par image ;
- les scores de confiance ;
- une vidéo annotée ;
- un fichier JSON ou Parquet.

Le critère de réussite principal n’est pas seulement une bonne détection image par image.

Le système doit surtout produire :

- des trajectoires cohérentes ;
- peu de changements d’identité ;
- un rappel élevé pour le ballon ;
- une stabilité suffisante sur les séquences longues.

Cette première brique ne constitue pas encore un assistant à l’arbitrage complet. Elle forme la couche de perception sur laquelle seront construits :

- la calibration du terrain ;
- la localisation métrique ;
- la reconnaissance des équipes ;
- la détection des contacts ;
- l’analyse du hors-jeu ;
- l’interprétation des règles.
