# Portefolio : Pipeline de Détection, Tracking et Analyse Vidéo :computer:

Pipeline de vision temps réel pour l'analyse de flux vidéo :
**détection**, **suivi multi-objets (MOT)** et **applications métier**
(comptage de personnes et temps de présence, détection de bagages abandonnés,
mesure de vitesse et comptage de véhicules). Le tout exposé via une **API web**
(FastAPI + Uvicorn) ou en locale (OpenCV).

> :mortar_board: Pierre Lefebvre - Doctorant DVRC / AID  
> :e-mail: contact : lefebvre.pierre0318@gmail.com  
> :handshake: linkedin : https://www.linkedin.com/in/pierre-lefebvre-29ab03205/  

---

## Aperçu

![Démo de l'API multi use-case](docs/demo_overview.gif)

> Sélection du use-case, analyse, dashboard et flux JSON temps réel.

Un même socle **détection + tracking**, trois applications métier, une interface unique :

| Cas d'usage | Description (en bref)                                                      |
|-------------|----------------------------------------------------------------------------|
| **Retail** | Comptage de personnes, temps de présence par zone (top 5)                  |
| **Bagages abandonnés** | Bagage immobile séparé de son propriétaire → alerte                        |
| **Trafic** | Vitesse des véhicules (homographie), comptage bi-directionnel par portique |

## Architecture

Structure modulaire en trois couches : on peut brancher n'importe quel
détecteur sur n'importe quel tracker, puis n'importe quelle logique métier
par-dessus - facilite l'évolutivité et la maintenabilité.

```
                  ┌────────────────┐
   flux vidéo ──► │    Détection   │    YOLO / YOLOX (ONNX)
                  └────────┬───────┘
                           │  détections (bbox + classes)
                  ┌────────▼───────┐
                  │    Tracking    │    ByteTrack · BoT-SORT · OC-SORT · SORT · C-BIoU
                  └────────┬───────┘
                           │  tracks (identités persistantes)
                  ┌────────▼────────┐
                  │ Logique métier  │   retail · luggage · trafic ...
                  └────────┬────────┘
                           │  frame annotée + métriques
                  ┌────────▼───────┐
                  │       API      │    FastAPI · stream MJPEG · dashboard
                  └────────────────┘
```

**Détection** : intégration de détecteurs YOLOX optimisés ONNX/GPU pour l'inférence temps réel.

**Tracking** : cinq algorithmes de suivi multi-objets state-of-the-art (ByteTrack, BoT-SORT, OC-SORT…) ré-implémentés à partir des articles et dépôts de référence (cf. [Références](#références)), permettant de choisir le meilleur compromis robustesse/vitesse selon le contexte.

**Logique métier** : transformation des trajectoires brutes en indicateurs exploitables (comptage, temps de présence, vitesse, détection d'événements).

**API** : analyse vidéo, tableau de bord et flux de données, le tout dans le navigateur.

---

## Démo

### :shopping_cart: Retail

<p align="left">
  <img src="assets/retail_demo.gif" alt="Démo du module retail analytics" width="960">
</p>

- Définition d’une zone d'intérêt polygonale au sol
- Détection et suivi des clients
- Calcul du temps de présence, persistant pour chaque track/personne
- Classement top5 des temps de présence par odre décroissant

### :briefcase: Bagages abandonnés

<p align="left">
  <img src="assets/luggage_demo.gif" alt="Démo du module de détection de bagages abandonnés" width="900">
</p>

- Suivi des personnes et des bagages
- Logique d’état : porté, immobile, séparé, abandonné
- Détection d’un bagage resté immobile et séparé de son propriétaire
- Déclenchement d’une alerte visuelle en cas d’abandon

### :vertical_traffic_light: Trafic

<p align="left">
  <img src="assets/trafic_demo.gif" alt="Démo du module trafic" width="900">
</p>

- Détection et suivi de véhicules
- Estimation de la vitesse par projection homographique image → route
- Lissage des vitesses avec filtre de Savitzky-Golay
- Comptage bidirectionnel des véhicules au passage de la ligne centrale

### Vous aimez les heatmaps ? Moi aussi :sunglasses:

<p align="left">
  <img src="assets/heatmap_retail.gif" alt="heatmap du usecase retail" width="900">
</p>

---

## Structure du dépôt

```
assets/         # images/gifs pour le README
src/
  detection/    # détecteurs ONNX + post-processing
  tracking/     # trackers MOT réimplémentés (ByteTrack, BoT-SORT, OC-SORT, SORT, C-BIoU)
  retail/       # comptage clients + temps de présence par zone
  luggage/      # détection de bagages abandonnés
  traffic/      # comptage + vitesses des véhicules
  api/          # FastAPI, démo web, registre de use cases
tools/          # zone_drawer, calibration homographie
tests/          # plusieurs tests unitaires par use case (non inclu)
configs/        # configuration use case (YAML) + zones/homographies (JSON)
```

---

## Installation

Créez puis activez l’environnement Conda :

```bash
conda create -n vision python=3.10
conda activate tracking_env
```

Installez ensuite les dépendances Python :

```bash
pip install -r requirements.txt
```

Ou créez directement l’environnement Conda depuis le fichier :

```bash
conda env create -f environment.yml
conda activate tracking_env
```

> **Note**  
> Les modèles et les données sont disponibles sur ce drive : TODO  
> Placer les modèles dans `models/`, les vidéos dans `data/` et les fichiers 
de configuration dans `configs/`.

---

## Utilisation

**Démo via l'API** (recommandé) :

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Ouvrir **http://localhost:8000**, choisir un use case, lancer l'analyse.

**Tests unitaires par use case** (fenêtre OpenCV) :

```bash
python tests/test_retail.py
python tests/test_traffic.py
python tests/test_luggage_monitor.py
```

NB: 
`Espace` = pause, `Q` = quitter  
debug prompté en console toutes les 100 frames.

**Définir une zone de surveillance (retail)** :

```bash
python tools/zone_drawer.py   # cliquez les sommets, ENTRÉE pour sauvegarder
```

**Réaliser une homographie (traffic / luggage)** :

```bash
python tools/homography_calibrator.py   # cliquez les sommets puis entrez les coordonnées (x, y) de chaque point
```

---

## Pistes d'évolution

### Général
- Conteneurisation Docker de l'API
- Fine-tuning des détecteurs sur le domaine cible pour améliorer le rappel
- Ajout de détecteurs transformer-based (RT-DETR)

### Amélioration des use-cases existants
- Retail : interactions avec les articles, heatmap de fréquentation, multi-zones
- Trafic : classification par type de véhicule, détection d'infractions

### Nouveaux use-cases
- Sécurité : détection de chute, port de matériel de sécurité, franchissement de zone interdite
- Parking : détection de places libres/occupées, temps de stationnement
- Football : attribution d'équipe, possession de balle, OCR sur les numéros des maillots


---

## Références

Trackers réimplémentés :

- **SORT** - Bewley et al., *Simple Online and Realtime Tracking* (2016) · [arXiv](https://arxiv.org/abs/1602.00763) · [github](https://github.com/abewley/sort)
- **ByteTrack** - Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (2022) · [arXiv](https://arxiv.org/abs/2110.06864) · [github](https://github.com/ifzhang/ByteTrack)
- **BoT-SORT** - Aharon et al., *BoT-SORT: Robust Associations Multi-Pedestrian Tracking* (2022) · [arXiv](https://arxiv.org/abs/2206.14651) · [github](https://github.com/NirAharon/BoT-SORT)
- **OC-SORT** - Cao et al., *Observation-Centric SORT* (2023) · [arXiv](https://arxiv.org/abs/2203.14360) · [github](https://github.com/noahcao/OC_SORT)
- **C-BIoU** - Yang et al., *Hard to Track Objects with Irregular Motions and Similar Appearances? Make It Easier by Buffering the Matching Space* (2023) · [arXiv](https://arxiv.org/abs/2211.14317)

Détecteurs :

- **YOLOX** - Ge et al. (2021) · [arXiv](https://arxiv.org/abs/2107.08430) · [github](https://github.com/Megvii-BaseDetection/YOLOX)
- **Ultralytics YOLO** · [github](https://github.com/ultralytics/ultralytics)
