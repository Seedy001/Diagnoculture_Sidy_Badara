# DiagnoCulture

Diagnostic des maladies foliaires de la tomate (PlantVillage, 10 classes).
Projet de fin de module — Apprentissage profond, UAM / Polytech Diamniadio.

## Données
- PlantVillage, sous-ensemble Tomate (10 classes : 9 maladies + saine).
- 16 011 images détectées ; 14 doublons exacts (pHash) retirés → 15 997 images.
- Source : Kaggle, dataset « emmarex/plantdisease », téléchargé automatiquement via `kagglehub`.
- Photos OOD (partie 4) : ~40 photos « non-feuille » dans `data/ood-photos/`.

## Installation
Projet prévu pour Google Colab (GPU T4 gratuit).
1. Ouvrir le notebook dans Colab, activer le GPU (Exécution → Modifier le type d'exécution → T4).
2. Monter Google Drive et placer le dossier `Project_clean/` (data, models, results) dans `MyDrive`.
3. Dépendances : `pip install -r requirements.txt` (le dataset est récupéré par `kagglehub` à l'exécution).

## Découpage (anti-fuite)
- Déduplication par perceptual hash avant le split.
- Split stratifié 70/15/15, seed = 42, figé dans `split.csv`.
- Jeu de test utilisé une seule fois (jamais pour régler un seuil/hyperparamètre).

## Reproductibilité
- SEED = 42 (random, numpy, torch, cuDNN déterministe).
- Normalisation calculée sur le train uniquement (`norm_stats.json`).
- Environnement : Google Colab (GPU T4).

## Ordre d'exécution (temps estimés sur T4)
1. `00_eda_split.ipynb` — EDA, dédup pHash, split, stats de normalisation (~10-15 min)
2. `01_cnn_from_scratch.ipynb` — CNN-Plain & CNN-Res, étude lr, gradients (~30-40 min)
3. `02_transfer_learning.ipynb` — MobileNetV2 (3 stratégies) + ResNet-50, Pareto (~20-25 min)
4. `03_autoencoders.ipynb` — auto-encodeur, étude de k, débruiteur, détection (~15-20 min)
5. `04_hybrid_open_set.ipynb` — système hybride de rejet de l'inconnu + OOD (~15 min)
6. `05_generative_extension.ipynb` — VAE (génération, interpolation, t-SNE) (~5-10 min)
7. `06_explicabilite_gradcam.ipynb` — Grad-CAM et analyse d'erreurs (~3-5 min)

## Structure du dépôt
```
diagnoculture/
├── README.md
├── requirements.txt
├── data/            # non versionné : dataset (kagglehub) + ood-photos/
├── notebooks/       # 00 à 06
├── models/          # poids .pt (CNN, transfer, AE, VAE)
├── results/         # courbes, matrices, figures, CSV
└── report/          # rapport PDF
```

## Usage d'IA générative
Assistance d'un agent IA (guidage + relecture du code), déclarée en annexe du rapport.
