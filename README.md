# FedPSPO: Federated Proximal Smoothing Policy Optimization for Medical Image Classification

> Reinforcement Learning-based Adaptive Federated Learning Framework for Privacy-Preserving Medical Image Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Overview

FedPSPO (**Federated Proximal Smoothing Policy Optimization**) is a reinforcement learning-driven federated learning framework that dynamically adapts federated optimization hyperparameters during training.

Unlike conventional federated learning approaches such as FedAvg, FedPSPO formulates federated optimization as a **Markov Decision Process (MDP)** and employs a **Policy-Smoothed PPO (PSPO)** agent to automatically optimize:

- Client learning rate
- Local training epochs
- Server learning rate
- Aggregation weights

This adaptive orchestration significantly improves convergence stability, feature consistency, and classification accuracy under heterogeneous (non-IID) federated environments.

The framework is evaluated on the **BloodMNIST** dataset from **MedMNIST v2** for blood cell classification.

---

## Highlights

- Reinforcement Learning-based Federated Orchestration
- Dynamic Hyperparameter Optimization
- Policy Smoothing for Stable PPO Updates
- Adaptive Client Aggregation
- ResNet-50 Backbone
- Non-IID Federated Training
- BloodMNIST Classification
- McNemar Statistical Validation
- Ablation Study
- Latent Feature Visualization

---

## Proposed Architecture

The FedPSPO pipeline consists of four major stages:

```
                +-------------------------+
                |    PPO Agent (Server)   |
                +-----------+-------------+
                            |
                State (Loss, Accuracy,
              Weight Similarity, etc.)
                            |
                    Continuous Actions
                            |
      --------------------------------------------
      |              |               |            |
   Client 1      Client 2        Client 3     Client K
      |              |               |            |
   Local SGD     Local SGD       Local SGD    Local SGD
      |              |               |            |
      --------------------------------------------
                    Adaptive Aggregation
                            |
                    Updated Global Model
                            |
                     Reward Calculation
                            |
                     PPO Policy Update
```

---

## Key Features

### Adaptive PPO Agent

The RL agent continuously learns the optimal federated strategy by observing

- Validation Loss
- Validation Accuracy
- Loss Reduction
- Accuracy Improvement
- Cosine Similarity between successive global models

---

### Adaptive Hyperparameters

FedPSPO automatically learns

- Client Learning Rate
- Local Epochs
- Server Learning Rate
- Aggregation Weights

instead of using manually fixed values.

---

### Policy Smoothing

Instead of abrupt PPO policy changes,

\[
\tilde{a}_t=\alpha a_{t-1}+(1-\alpha)a_t
\]

is used to stabilize learning and reduce client drift.

---

### Reward Function

The PPO agent optimizes

\[
r_t=\beta_1A_t-\beta_2L_t+\beta_3C_t
\]

where

- Accuracy
- Validation Loss
- Cosine Similarity

jointly determine the reward.

---

## Dataset

The experiments use

**BloodMNIST (MedMNIST v2)**

- 17,092 RGB images
- 8 blood cell classes
- Image size: 224 × 224

Classes:

- Basophil
- Eosinophil
- Erythroblast
- Immature Granulocyte
- Lymphocyte
- Monocyte
- Neutrophil
- Platelet

---

## Repository Structure

```
FedPSPO/
│
├── FedPSPO_Code.py          # Main training pipeline
├── README.md
├── outputs/
│   ├── models/
│   ├── figures/
│   ├── checkpoints/
│   └── logs/
│
├── images/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   ├── feature_maps.png
│   └── loss_curves.png
│
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/USERNAME/FedPSPO.git

cd FedPSPO
```

Create environment

```bash
conda create -n fedpspo python=3.10

conda activate fedpspo
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- TorchVision
- NumPy
- Scikit-Learn
- Matplotlib
- tqdm
- statsmodels

---

## Training

Run

```bash
python FedPSPO_Code.py
```

The script automatically

- Loads BloodMNIST
- Creates federated clients
- Initializes the PPO agent
- Trains the global model
- Updates policy
- Saves the trained model
- Generates plots

---

## Configuration

Important parameters are located inside

```python
class CFG
```

Example

```python
NUM_CLIENTS = 4
NUM_ROUNDS = 40
BATCH_SIZE = 64
IMG_SIZE = 224
RL_LR = 1e-3
REWARD_WIN = 5
```

---

## Model Components

### Global Model

- ResNet-50
- Dropout Regularization
- Custom Classification Head

### RL Agent

- Gaussian Policy Network
- Continuous Action Space
- Policy Smoothing
- REINFORCE Update

### Federated Server

- Adaptive Aggregation
- Weighted Pseudo Gradient
- Dynamic Server Learning Rate

---

## Experimental Results

| Metric | FedPSPO |
|----------|---------|
| Test Accuracy | **97.87%** |
| AUC | **99.8%** |
| Macro Recall | 97.03% |
| Macro F1-score | 97.04% |

FedPSPO consistently outperformed

- FedAvg
- HQCNN
- MCFLM-CB
- ResNet-18
- AutoML
- MedViT-L

---

## Visualizations

The implementation includes

- RL Loss Curves
- Validation Loss
- Accuracy Curves
- Confusion Matrix
- Latent Feature Maps
- Mean Activation Maps
- Ablation Study
- McNemar Test

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{FedPSPO2026,
  title={FedPSPO: Enhancing Cytology Image Classification through Federated Learning-Based Reinforcement-Driven Hyperparameter Adaptation},
  author={Your Name},
  booktitle={Proceedings of ICVGIP},
  year={2026}
}
```

---

## Future Work

- Asynchronous Federated Learning
- Personalized Federated Learning
- Multi-Agent Reinforcement Learning
- Differential Privacy Integration
- Secure Aggregation
- Cross-Silo Healthcare Deployment

---

## License

This repository is released under the MIT License.

---

## Acknowledgements

- MedMNIST
- PyTorch
- OpenAI
- PPO (Schulman et al.)
- Federated Learning (McMahan et al.)

---

## Contact

For questions, collaborations, or research discussions, please open an issue or contact the repository maintainer.
