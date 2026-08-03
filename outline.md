# Outline du projet referAI

## 1. Vision du projet

L'objectif final est de construire un assistant d'arbitrage capable de suivre un match de
football, de représenter les joueurs comme des agents et de détecter des événements tels que
les hors-jeu, les contacts et les fautes potentielles.

Le système doit rester explicable. Il doit séparer clairement :

1. les observations visuelles produites par les modèles ;
2. les états et événements déduits de ces observations ;
3. les règles qui transforment ces événements en décisions arbitrales proposées.

Une décision devra toujours être accompagnée de ses preuves : images concernées, agents
impliqués, positions, incertitudes et règle appliquée.

## 2. Utilisation de ce document

Ce fichier est la feuille de route vivante du projet. Une seule étape doit être marquée
`EN COURS` à la fois.

Statuts possibles :

- `A FAIRE` : aucun travail validé ;
- `EN COURS` : étape active ;
- `BLOQUEE` : dépendance ou décision manquante ;
- `LIVREE` : définition de fini entièrement satisfaite.

A la fin de chaque étape :

1. exécuter les tests et évaluations prévus ;
2. renseigner le bilan de livraison de l'étape ;
3. ajouter les métriques et les chemins des artefacts ;
4. noter les limites et dettes techniques restantes ;
5. mettre à jour le statut ;
6. désigner explicitement la prochaine étape.

Les objectifs chiffrés qui ne sont pas encore connus devront être fixés après la première
baseline mesurée. Ils ne doivent pas être inventés sans données expérimentales.

## 3. Vue d'ensemble

```text
Vidéo du match
  -> Détection et tracking
  -> Sélection des plans exploitables
  -> Calibration du terrain
  -> Agents localisés en coordonnées métriques
  -> Equipes, rôles et sens d'attaque
  -> Etat du ballon et événements élémentaires
  -> Détection du hors-jeu
  -> Détection des contacts et fautes potentielles
  -> Moteur de règles explicable
  -> Rapport arbitral avec preuves et incertitudes
```

| Etape | Module | Statut initial | Dépend de |
|---|---|---|---|
| 0 | Perception et tracking fiables | EN COURS | - |
| 1 | Segmentation des plans caméra | A FAIRE | 0 |
| 2 | Calibration et coordonnées terrain | A FAIRE | 1 |
| 3 | Modèle d'agents et état du match | A FAIRE | 0, 2 |
| 4 | Equipes, rôles et sens d'attaque | A FAIRE | 3 |
| 5 | Ballon, possession et événements élémentaires | A FAIRE | 3, 4 |
| 6 | Détection du hors-jeu | A FAIRE | 2, 4, 5 |
| 7 | Contacts et fautes potentielles | A FAIRE | 3, 5 |
| 8 | Moteur de règles et explicabilité | A FAIRE | 6, 7 |
| 9 | Evaluation de bout en bout et exploitation | A FAIRE | 8 |

## 4. Principes transverses

### 4.1 Séparation des niveaux de vérité

Chaque donnée doit indiquer son origine :

- `observed` : directement produite par la détection ou une annotation ;
- `estimated` : reconstruite, lissée ou interpolée ;
- `inferred` : déduite par un modèle d'événement ;
- `ruled` : conclusion d'un moteur de règles.

### 4.2 Gestion de l'incertitude

Chaque transformation importante doit produire une confiance ou une marge d'erreur. Un cas
ambigu doit pouvoir devenir `review_required` plutôt qu'une décision artificiellement sûre.

### 4.3 Prévention des fuites de données

Les ensembles d'entraînement, de validation et de test doivent être séparés par match, et si
possible par compétition ou stade. Deux clips issus du même match ne doivent pas se retrouver
dans deux partitions différentes.

### 4.4 Reproductibilité

Chaque expérience doit conserver :

- la configuration utilisée ;
- la version du code et des poids ;
- la liste des matchs évalués ;
- les métriques par match et agrégées ;
- les artefacts visuels nécessaires à l'analyse des erreurs.

## 5. Etape 0 - Perception et tracking fiables

**Statut : `EN COURS`**

### Objectif

Produire des trajectoires stables pour les joueurs, gardiens, arbitres et le ballon à partir
d'une vidéo ou d'une séquence MOT.

### Etat déjà implémenté

- préparation de SportsMOT et SoccerNet au format YOLO ;
- fine-tuning YOLO11 ;
- tracking ByteTrack ;
- export JSON, JSONL, Parquet, MOT et trajectoires ;
- génération d'une vidéo annotée ;
- profils matériels K80, RTX 3090 Ti et CPU ;
- augmentations et sur-échantillonnage des classes rares ;
- export compatible avec TrackEval ;
- évaluation automatisée du tracking avec HOTA, DetA, AssA, MOTA, MOTP, IDF1,
  ID switches, fragmentations, rappel et précision.

### Travaux restants

- constituer ou intégrer des annotations pour `goalkeeper`, `referee` et `ball` ;
- entraîner et évaluer le détecteur quatre classes ;
- régler les seuils ByteTrack sur le split de validation ;
- mesurer le rappel du ballon par taille apparente ;
- tester des séquences longues, occultations, regroupements et changements d'échelle ;
- intégrer une ReID si les changements d'identité restent trop nombreux ;
- permettre l'inférence directe sur une séquence d'images si ce format devient nécessaire.

### Livrables attendus

- poids du meilleur détecteur ;
- configuration finale du tracker ;
- rapport de métriques détection et tracking ;
- vidéos annotées représentatives des succès et échecs ;
- schéma stable des observations et trajectoires ;
- commande reproductible d'évaluation complète.

### Définition de fini

- les quatre classes utiles sont prises en charge ou une limitation de périmètre est actée ;
- les métriques sont mesurées sur un test séparé par match ;
- les objectifs chiffrés retenus sont atteints ;
- les sorties longues sont générées sans conserver toute la vidéo en mémoire ;
- les erreurs typiques sont documentées ;
- les tests automatisés du module passent.

### Bilan de livraison

- Date :
- Statut final :
- Fonctionnalités livrées :
- Métriques obtenues :
- Artefacts et poids :
- Tests exécutés :
- Limites connues :
- Décisions prises :

### Journal d'avancement

| Date | Livraison intermédiaire | Validation | Suite |
|---|---|---|---|
| 2026-08-03 | Commande `evaluate-tracking`, inférence par séquence MOT, intégration TrackEval et export JSON des métriques | Tests automatisés et scénario synthétique parfait HOTA/MOTA/IDF1 = 100 | Mesurer sur `val` puis régler les seuils ByteTrack |
- Prochaine étape :

## 6. Etape 1 - Segmentation des plans caméra

**Statut : `A FAIRE`**

### Objectif

Empêcher les gros plans, ralentis, replays et transitions de contaminer la reconstruction du
match. Identifier les segments sur lesquels une localisation terrain est possible.

### Travaux

- détecter les changements de plan ;
- classer les plans en vue tactique, vue rapprochée, derrière le but, replay ou inexploitable ;
- détecter les périodes de replay et éviter de les mélanger avec la timeline réelle ;
- attribuer un identifiant de segment et de caméra ;
- propager un indicateur `camera_valid` aux étapes suivantes ;
- reprendre proprement le tracking après une coupure de caméra sans fusionner arbitrairement
  les identités.

### Livrables attendus

- module proposé : `src/referai/scene.py` ;
- configuration des seuils et classes caméra ;
- timeline JSON des segments ;
- visualisation des frontières de plans ;
- dataset annoté de plans représentatifs.

### Métriques

- précision, rappel et F1 par type de plan ;
- erreur temporelle sur les frontières ;
- taux de replays injectés à tort dans la timeline ;
- taux de frames tactiques rejetées à tort.

### Définition de fini

- les plans non exploitables et replays sont filtrés avec les objectifs validés ;
- chaque frame reçoit un segment, un type de plan et une confiance ;
- les coupures ne créent pas de continuité temporelle mensongère ;
- les tests et une démonstration sur plusieurs matchs sont disponibles.

### Bilan de livraison

- Date :
- Statut final :
- Fonctionnalités livrées :
- Métriques obtenues :
- Artefacts :
- Tests exécutés :
- Limites connues :
- Prochaine étape :

## 7. Etape 2 - Calibration et coordonnées terrain

**Statut : `A FAIRE`**

### Objectif

Transformer les coordonnées image en coordonnées métriques sur un terrain standard afin de
mesurer distances, alignements et positions de hors-jeu.

### Travaux

- détecter lignes, intersections, cercle central, surfaces et points de penalty ;
- associer les primitives détectées à un modèle géométrique du terrain ;
- estimer une homographie image-terrain par frame ou image clé ;
- lisser la calibration au cours d'un même plan caméra ;
- détecter les calibrations invalides ou insuffisamment contraintes ;
- projeter au sol le point de contact de chaque joueur, initialement le milieu du bas de la
  boîte ;
- projeter le ballon en tenant compte du fait qu'il peut être en hauteur ;
- propager une erreur de calibration en pixels et en mètres.

### Livrables attendus

- module proposé : `src/referai/calibration.py` ;
- modèle paramétrique du terrain ;
- homographie et confiance par frame ;
- export des positions `(x, y)` en mètres ;
- vue aérienne 2D permettant de contrôler visuellement les projections.

### Métriques

- erreur de reprojection en pixels ;
- erreur de localisation en mètres sur des points annotés ;
- stabilité temporelle de l'homographie ;
- taux de frames calibrables ;
- erreur près des lignes critiques pour le hors-jeu.

### Définition de fini

- une position pixel peut être projetée dans un référentiel terrain documenté ;
- l'erreur est mesurée et attachée à chaque projection ;
- les mouvements et zooms de caméra sont gérés au sein d'un segment ;
- une calibration douteuse bloque une décision géométrique automatique ;
- les projections sont vérifiées visuellement et par tests numériques.

### Bilan de livraison

- Date :
- Statut final :
- Fonctionnalités livrées :
- Métriques obtenues :
- Artefacts :
- Tests exécutés :
- Limites connues :
- Prochaine étape :

## 8. Etape 3 - Modèle d'agents et état du match

**Statut : `A FAIRE`**

### Objectif

Transformer les tracks en agents persistants et produire un état cohérent du match à chaque
instant.

### Schémas cibles

```text
AgentState
  track_id, classe, équipe, rôle
  position_m, vitesse_m_s, accélération_m_s2, direction
  bbox, visible, confiance
  source: observed | estimated | inferred

BallState
  position_image, position_terrain, vitesse, porteur_probable
  visible, aérien_probable, confiance

GameState
  frame_id, timestamp, segment_id, camera_valid
  agents[], ball, direction_attaque, événements[]
```

### Travaux

- définir des dataclasses et un schéma versionné ;
- lisser positions, vitesses et accélérations ;
- interpoler uniquement les absences courtes avec une confiance décroissante ;
- distinguer absence, occultation, interpolation et observation réelle ;
- calculer distances, angles et voisinages entre agents ;
- fournir une API temporelle pour obtenir l'état à une frame ou sur une fenêtre ;
- garantir que les étapes suivantes ne dépendent plus directement des objets Ultralytics.

### Livrables attendus

- modules proposés : `src/referai/agents.py` et `src/referai/game_state.py` ;
- schéma JSON versionné ;
- convertisseur depuis les sorties du tracker ;
- visualisation 2D des agents et de leurs trajectoires ;
- tests sur trajectoires synthétiques et réelles.

### Définition de fini

- chaque track valide devient un agent avec un état temporel ;
- les unités et référentiels sont explicites ;
- les données estimées ne sont jamais confondues avec les observations ;
- les relations spatiales nécessaires aux événements sont disponibles ;
- le schéma reste lisible sans Ultralytics ou ByteTrack.

### Bilan de livraison

- Date :
- Statut final :
- Fonctionnalités livrées :
- Schéma livré :
- Métriques obtenues :
- Tests exécutés :
- Limites connues :
- Prochaine étape :

## 9. Etape 4 - Equipes, rôles et sens d'attaque

**Statut : `A FAIRE`**

### Objectif

Attribuer une équipe et un rôle stable à chaque agent, puis déterminer quelle équipe attaque
dans quelle direction pour chaque période du match.

### Travaux

- extraire une représentation visuelle du maillot à partir des crops joueurs ;
- regrouper les agents en deux équipes et isoler arbitres et gardiens ;
- stabiliser l'étiquette d'équipe sur toute la trajectoire par vote temporel ;
- gérer les maillots visuellement proches, ombres et changements de luminosité ;
- identifier les gardiens avec leur apparence et leur zone habituelle ;
- détecter le sens d'attaque et les changements de côté ;
- exposer la confiance et une classe `unknown` en cas d'ambiguïté.

### Livrables attendus

- module proposé : `src/referai/teams.py` ;
- modèle ou méthode de classification documentée ;
- enrichissement de `AgentState` avec `team_id` et `role` ;
- sens d'attaque dans `GameState` ;
- galerie d'erreurs et matrice de confusion.

### Métriques

- exactitude d'équipe par track, et non seulement par image ;
- F1 pour joueurs, gardiens et arbitres ;
- taux de changement erroné d'équipe au sein d'un track ;
- exactitude du sens d'attaque.

### Définition de fini

- les deux équipes sont séparées de façon stable ;
- les arbitres ne sont pas affectés à une équipe ;
- le sens d'attaque est connu avec une confiance exploitable ;
- les cas inconnus sont conservés comme tels ;
- les métriques sont mesurées sur plusieurs matchs et équipements différents.

### Bilan de livraison

- Date :
- Statut final :
- Fonctionnalités livrées :
- Métriques obtenues :
- Artefacts :
- Tests exécutés :
- Limites connues :
- Prochaine étape :

## 10. Etape 5 - Ballon, possession et événements élémentaires

**Statut : `A FAIRE`**

### Objectif

Reconstruire l'état temporel du ballon et détecter les événements atomiques nécessaires au
raisonnement arbitral.

### Travaux

- lisser la trajectoire du ballon sans effacer les changements de direction réels ;
- interpoler les occultations courtes avec une incertitude croissante ;
- détecter les touches de balle à partir de la distance, du mouvement et de la pose ;
- attribuer une possession probable à un agent et à une équipe ;
- détecter passe, réception, interception, tir et perte de possession ;
- détecter les sorties de balle et la ligne franchie ;
- estimer si le ballon est probablement au sol ou aérien ;
- produire un intervalle temporel et une confiance pour chaque événement.

### Livrables attendus

- modules proposés : `src/referai/ball.py` et `src/referai/events.py` ;
- timeline d'événements élémentaires ;
- clips vidéo de diagnostic autour de chaque événement ;
- annotations spécifiques aux touches et passes ;
- visualisation synchronisée vidéo, terrain 2D et timeline.

### Métriques

- précision et rappel des touches, passes, tirs et sorties ;
- erreur temporelle en frames ou millisecondes ;
- exactitude du possesseur et de l'équipe en possession ;
- erreur de trajectoire pendant les occultations ;
- taux d'événements avec acteur inconnu.

### Définition de fini

- l'instant de la dernière touche est estimé avec une incertitude ;
- le joueur et l'équipe concernés sont identifiés lorsque les preuves suffisent ;
- les événements sont persistés dans un format versionné ;
- les sorties de balle sont rattachées à une ligne du terrain ;
- l'évaluation est effectuée sur des séquences annotées indépendantes.

### Bilan de livraison

- Date :
- Statut final :
- Fonctionnalités livrées :
- Métriques obtenues :
- Artefacts :
- Tests exécutés :
- Limites connues :
- Prochaine étape :

## 11. Etape 6 - Détection du hors-jeu

**Statut : `A FAIRE`**

### Objectif

Détecter une position de hors-jeu à l'instant de la passe, puis déterminer si le joueur
concerné participe activement à l'action.

### Pipeline fonctionnel

1. détecter l'instant exact de la passe ou de la touche pertinente ;
2. identifier le passeur et l'équipe attaquante ;
3. récupérer les positions métriques et incertitudes à cet instant ;
4. identifier l'avant-dernier adversaire sur l'axe d'attaque ;
5. comparer l'attaquant au ballon et à la ligne de hors-jeu ;
6. vérifier qu'il se trouve dans la moitié adverse ;
7. suivre la suite de l'action pour déterminer son implication active ;
8. rendre `onside`, `offside`, `review_required` ou `not_applicable`.

### Travaux

- définir précisément l'axe d'attaque dans le référentiel terrain ;
- utiliser la partie du corps pertinente, avec pose ou segmentation pour les cas serrés ;
- propager les erreurs de calibration, de pose et de timing ;
- distinguer position de hors-jeu et infraction de hors-jeu ;
- gérer les remises en jeu et situations où la règle n'est pas applicable ;
- générer une image de preuve avec ligne, positions et marges ;
- conserver la version de la règle appliquée.

### Livrables attendus

- module proposé : `src/referai/offside.py` ;
- dataset de situations annotées à l'instant de la passe ;
- décision structurée avec marge en mètres et intervalle d'incertitude ;
- overlay de la ligne de hors-jeu et clip de preuve ;
- rapport d'erreurs séparant timing, calibration, tracking et règle.

### Métriques

- précision, rappel et F1 des infractions ;
- erreur sur l'instant de passe ;
- erreur de position de la ligne en mètres ;
- exactitude de l'avant-dernier adversaire ;
- taux de décisions `review_required` ;
- métriques séparées pour cas nets et cas serrés.

### Définition de fini

- la position et l'implication sont évaluées séparément ;
- chaque décision possède une preuve visuelle et numérique ;
- les décisions trop proches de l'incertitude deviennent `review_required` ;
- les règles non applicables sont explicitement traitées ;
- les erreurs sont traçables jusqu'au module responsable.

### Bilan de livraison

- Date :
- Statut final :
- Fonctionnalités livrées :
- Métriques obtenues :
- Artefacts :
- Tests exécutés :
- Limites connues :
- Prochaine étape :

## 12. Etape 7 - Contacts et fautes potentielles

**Statut : `A FAIRE`**

### Objectif

Détecter les interactions physiques pertinentes, puis distinguer un simple contact d'une
faute potentielle sans confondre observation et décision réglementaire.

### Travaux

- générer des candidats avec distance, convergence, vitesse relative et changements brutaux ;
- détecter chutes, tacles, sauts, charges et contacts probables ;
- intégrer une estimation de pose pour localiser les parties du corps ;
- analyser une fenêtre temporelle avant et après le contact ;
- prendre en compte le ballon, sa jouabilité et l'ordre des contacts ;
- classifier d'abord `no_contact`, `legal_contact`, `potential_foul` et `uncertain` ;
- enrichir ensuite avec tacle, charge, obstruction, main et autres catégories validées ;
- produire les agents impliqués, l'instant, la zone et les preuves.

### Livrables attendus

- modules proposés : `src/referai/contacts.py` et `src/referai/fouls.py` ;
- dataset temporel annoté autour des interactions ;
- générateur de candidats à rappel élevé ;
- classifieur temporel des contacts ;
- clips avant/après et vue terrain des agents impliqués.

### Métriques

- rappel du générateur de candidats ;
- précision, rappel et F1 par type de contact ;
- faux positifs par minute de match ;
- erreur temporelle de l'instant de contact ;
- performance selon visibilité, distance caméra et occultation.

### Définition de fini

- les candidats couvrent la majorité des contacts annotés ;
- un contact visuel n'est pas automatiquement déclaré faute ;
- les cas ambigus sont signalés ;
- la décision expose les agents, le ballon et le contexte temporel ;
- les performances sont mesurées par scénario et non seulement en moyenne globale.

### Bilan de livraison

- Date :
- Statut final :
- Fonctionnalités livrées :
- Métriques obtenues :
- Artefacts :
- Tests exécutés :
- Limites connues :
- Prochaine étape :

## 13. Etape 8 - Moteur de règles et explicabilité

**Statut : `A FAIRE`**

### Objectif

Transformer des observations et événements en propositions de décisions auditables, sans
cacher la logique réglementaire dans un modèle opaque.

### Travaux

- définir un schéma commun `Evidence -> Event -> RuleDecision` ;
- implémenter des règles versionnées et testables ;
- distinguer fait observé, hypothèse, condition réglementaire et conclusion ;
- propager les confiances et marges d'erreur ;
- définir les seuils d'acceptation, rejet et revue humaine ;
- conserver la trace complète des entrées ayant mené à la décision ;
- générer un rapport lisible et des overlays synchronisés ;
- prévoir la mise à jour des règles sans réentraîner les modèles de perception.

### Livrables attendus

- module proposé : `src/referai/rules.py` ;
- catalogue versionné des règles prises en charge ;
- moteur déterministe avec tests unitaires par cas réglementaire ;
- format de décision JSON ;
- rapport HTML ou vidéo réunissant décision, preuve et incertitude.

### Définition de fini

- chaque conclusion référence une règle et ses preuves ;
- la même entrée et la même version de règle produisent la même sortie ;
- les données manquantes provoquent une abstention explicite ;
- les cas limites sont couverts par des tests réglementaires ;
- la décision peut être auditée sans examiner le code du modèle.

### Bilan de livraison

- Date :
- Statut final :
- Règles livrées :
- Métriques obtenues :
- Artefacts :
- Tests exécutés :
- Limites connues :
- Prochaine étape :

## 14. Etape 9 - Evaluation de bout en bout et exploitation

**Statut : `A FAIRE`**

### Objectif

Valider le système sur des matchs complets, mesurer les erreurs en cascade et préparer une
utilisation reproductible avec revue humaine.

### Travaux

- construire un benchmark figé de matchs complets ;
- mesurer chaque module séparément et la chaîne complète ;
- attribuer les erreurs finales à la détection, au tracking, à la calibration, aux événements
  ou aux règles ;
- mesurer latence, mémoire, FPS et stabilité ;
- définir un mode différé puis, si nécessaire, un mode quasi temps réel ;
- fournir une interface de revue des événements candidats ;
- journaliser les versions des modèles, configurations et règles ;
- tester robustesse, reprise après erreur et traitement de longues vidéos.

### Métriques de bout en bout

- précision et rappel des décisions arbitrales par type ;
- faux positifs par match ;
- taux d'abstention et de revue humaine ;
- calibration des scores de confiance ;
- délai entre l'action et la proposition ;
- disponibilité du pipeline sur un match complet ;
- distribution des causes d'erreur par module.

### Définition de fini

- un match complet peut être traité de manière reproductible ;
- chaque événement proposé est accompagné de preuves ;
- les performances et limites sont documentées par scénario ;
- les versions des modèles, données et règles sont traçables ;
- les critères d'usage automatique ou avec revue humaine sont explicitement fixés.

### Bilan de livraison

- Date :
- Statut final :
- Fonctionnalités livrées :
- Métriques finales :
- Matchs du benchmark :
- Artefacts :
- Tests exécutés :
- Limites connues :
- Suite décidée :

## 15. Registre des décisions

Ajouter ici les décisions qui modifient l'architecture, les données ou les objectifs.

| Date | Etape | Décision | Justification | Conséquences |
|---|---|---|---|---|
| A renseigner | 0 | Utiliser YOLO11 et ByteTrack comme baseline | Pipeline existant | Réévaluer après les métriques de tracking |

## 16. Journal global des livraisons

Ajouter une ligne lorsqu'une étape est officiellement livrée.

| Date | Etape livrée | Version ou commit | Résumé | Métriques clés | Prochaine étape |
|---|---|---|---|---|---|
| - | - | - | - | - | - |
