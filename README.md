# FedPSPO: Federated Proximal Smoothing Policy Optimization for Medical Image Classification

> Reinforcement learning-based adaptive federated learning for privacy-preserving medical image classification

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-GPU-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Overview

FedPSPO (**Federated Proximal Smoothing Policy Optimization**) is a federated learning framework that adapts its own hyperparameters during training.

When client data are non-IID, fixed hyperparameters such as those in FedAvg often cause client drift and unstable convergence. FedPSPO treats each communication round as a step of a **Markov decision process (MDP)**. A **PPO agent with temporal action smoothing** reads server-side validation feedback and sets, in every round:

- the client learning rate `lr`
- the number of local epochs `E`
- the server aggregation weight `λ`
- the number of selected features `N`
- the number of client clusters `K`

All actions are bounded, so FedPSPO keeps the non-convex convergence guarantee of FedAvg with two-sided learning rates, whatever the policy learns.

FedPSPO is evaluated on three **MedMNIST v2** datasets (**BloodMNIST**, **DermaMNIST**, **PneumoniaMNIST**) against **FedAvg**, **FedProx** and **SCAFFOLD** under an identical protocol.

---

## Highlights

- Federated optimization cast as an MDP, solved with PPO (clipped objective, actor-critic, GAE)
- Temporal policy smoothing, so hyperparameters change gradually between rounds
- Multi-objective reward: accuracy, loss and global-model stability
- Feature-based client clustering for aggregation (top-`N` features, `K` clusters)
- Bounded actions that preserve the FedAvg convergence guarantee
- ResNet-18 backbone at 224 × 224
- Non-IID clients (Dirichlet label skew)
- Three datasets, three seeds, three FL baselines
- Ablation study (7 variants) and non-IID sweep (Dirichlet α = 0.1, 0.5, 1.0)
- About 30% lower wall-clock time per round than the baselines, with less than 0.15 s of server-side overhead per round

---

## Method

```
                 +------------------------------------+
                 |        PPO agent (server)          |
                 |  state s_t -> action a_t -> smooth |
                 +-----------------+------------------+
                                   |
               smoothed (lr, E, λ, N, K) for round t
                                   |
      -------------------------------------------------------
      |               |                |                    |
  Client 1        Client 2         Client 3             Client 4
  local SGD       local SGD        local SGD            local SGD
  (lr, E)         (lr, E)          (lr, E)              (lr, E)
      |               |                |                    |
      -------- model updates + feature embeddings ----------
                                   |
         Feature selection (top-N) + K-means clustering of clients
                                   |
           Cluster-weighted aggregation with server weight λ
                                   |
                Updated global model, validated on server
                                   |
                  Reward r_t  ->  PPO policy update
```

### State

At the start of round `t` the agent observes

```
s_t = [ L_val, A_val, ΔL, ΔA, cos(W_t, W_{t-1}) ]
```

that is, the validation loss and accuracy, their change since the last round, and the cosine similarity between successive global models.

### Actions and bounds

| Action | Meaning | Range | Initial |
|---|---|---|---|
| `lr` | client learning rate (log scale) | 1e-4 to 1e-1 | 0.01 |
| `E` | local epochs | 1 to 5 | 5 |
| `λ` | server aggregation weight | 0.05 to 1.5 | 0.1 |
| `N` | selected embedding features | 32 to 512 | 256 |
| `K` | client clusters | 1 to number of clients | 2 |

### Policy smoothing

```
ã_t = α · a_{t-1} + (1 − α) · a_t ,   α = 0.25
```

### Reward

```
r_t = β1 · A_val − β2 · L_val + β3 · C_t ,   (β1, β2, β3) = (1, 1, 0.5)
```

where `C_t` is the cosine similarity between successive global models.

### Aggregation

```
W_{t+1} = W_t + λ · Σ_c (1/|C|) Σ_{k∈c} (n_k / n_c) ΔW_k
```

With `K = 1` this reduces to FedAvg weighting.

---

## Datasets

All three come from **MedMNIST v2**. The 224 × 224 versions are used.

| Dataset | Task | Classes | Images |
|---|---|---|---|
| BloodMNIST | blood cell microscopy | 8 | 17,092 |
| DermaMNIST | dermatoscopy | 7 | 10,015 |
| PneumoniaMNIST | chest X-ray, pneumonia vs normal | 2 | 5,856 |

The training set is split across **4 clients** with a **Dirichlet (α = 0.5)** label-skew partition. The official validation split is used on the server for the reward, and the official test split is used for evaluation only.

---

## Repository structure

```
FedPSPO/
├── FedPSPO.ipynb      # all experiments (3 cells, see below)
├── README.md
├── requirements.txt
└── images/            # figures from the paper (optional)
```

`FedPSPO.ipynb` has three code cells:

| Cell | What it does | Output |
|---|---|---|
| 1. Main experiments | Seed 42: 4 methods × 3 datasets, 7 ablations, Dirichlet 0.1 / 1.0 sweep, tables and figures | one JSON per run, `SUMMARY.txt`, figures |
| 2. Extra seeds | Seeds 7 and 123 for the 12 main runs; mean ± std over the three seeds and win counts | `*_s7.json`, `*_s123.json`, `SUMMARY_SEEDS.txt` |
| 3. Curves figure | Test accuracy per round, mean over seeds, all methods and datasets | `curves_all.pdf`, `curves_all.png` |

All outputs are written to `/kaggle/working/fedpspo_outputs/`.

---

## How to run

The notebook was developed and run on **Kaggle with a GPU**.

1. Download the MedMNIST v2 files `bloodmnist_224.npz`, `dermamnist_224.npz` and `pneumoniamnist_224.npz` from [medmnist.com](https://medmnist.com/) (Zenodo), and attach them to the notebook as a Kaggle dataset.
2. Run the cells in order.
3. Finished runs are skipped automatically. If the session time limit is reached (`TIME_BUDGET_H`), save the version, attach its output as an input, and run again. The notebook continues where it stopped.

To run outside Kaggle, change `OUT` and the `/kaggle/input/` search path in `find_npz` to local folders.

### Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
torch>=2.0
torchvision
numpy
scikit-learn
statsmodels
matplotlib
```

---

## Configuration

The main settings are at the top of cell 1:

```python
SEED = 42; NUM_CLIENTS = 4; ROUNDS = 40; BATCH = 64; IMG = 224
INIT   = dict(lr=0.01, E=5, lam=0.1, N=256, K=2)
BOUNDS = dict(lr=(1e-4, 1e-1), E=(1, 5), lam=(0.05, 1.5), N=(32, 512))
PPO    = dict(clip=0.2, epochs=4, buf=4, gamma=0.99, gae=0.95, lr=3e-4)
SMOOTH = 0.25; BETA = (1.0, 1.0, 0.5); PROX_MU = 0.01
```

---

## Model components

**Global model:** ResNet-18 trained from scratch, with a linear classification head.

**PPO agent:** actor-critic MLPs (2 × 64, tanh), Gaussian policy over 5 continuous actions, clipped PPO objective with GAE, temporal action smoothing.

**Server:** feature-based client clustering (top-`N` embedding features by variance, K-means with `K` clusters), cluster-weighted aggregation, adaptive server weight `λ`.

**Baselines:** FedAvg, FedProx (μ = 0.01) and SCAFFOLD, all using the same backbone, clients, rounds and data split.

---

## Results

Test results, mean ± std over seeds 42, 7 and 123. ResNet-18, 4 clients, Dirichlet α = 0.5, 40 rounds, final-round model (no checkpoint selection).

| Dataset | Method | Accuracy (%) | AUC (%) | s / round |
|---|---|---|---|---|
| BloodMNIST | FedAvg | 96.03 ± 2.96 | 99.83 ± 0.14 | 98.0 |
| | FedProx | 97.38 ± 1.54 | 99.90 ± 0.03 | 99.0 |
| | SCAFFOLD | 96.14 ± 1.90 | 99.81 ± 0.13 | 97.9 |
| | **FedPSPO** | **97.74 ± 1.42** | **99.90 ± 0.04** | **65.5** |
| DermaMNIST | FedAvg | 73.23 ± 3.33 | 89.64 ± 2.30 | 60.9 |
| | FedProx | 74.16 ± 2.12 | 89.58 ± 1.88 | 61.4 |
| | SCAFFOLD | 72.17 ± 2.28 | 90.40 ± 1.94 | 59.0 |
| | **FedPSPO** | **75.48 ± 1.20** | **92.07 ± 0.47** | **41.0** |
| PneumoniaMNIST | FedAvg | 78.74 ± 11.01 | 94.27 ± 0.65 | 36.3 |
| | FedProx | 79.65 ± 12.13 | 93.44 ± 3.00 | 37.1 |
| | SCAFFOLD | 82.59 ± 8.21 | **95.71 ± 0.99** | 35.8 |
| | **FedPSPO** | **83.60 ± 7.75** | 93.12 ± 3.02 | **25.3** |

- FedPSPO has the highest mean accuracy on all three datasets and beats FedAvg on every seed on BloodMNIST and PneumoniaMNIST.
- Its margin over FedProx on BloodMNIST is small (+0.36 points).
- All methods are unstable on PneumoniaMNIST. Accuracy varies a lot across seeds while AUC stays high, because the decision threshold shifts under extreme binary label skew. SCAFFOLD has the best AUC there.
- On DermaMNIST, FedPSPO matches the centralized ResNet-18 at the same input size (75.48% vs 75.4%) without sharing images.

---

## Visualizations

The notebook produces:

- test-accuracy curves per round for all methods (`curves_all.pdf`)
- per-dataset training curves (`<dataset>_curves.pdf`)
- the ablation bar chart (`bloodmnist_ablation.pdf`)
- confusion matrices (`<dataset>_confusion.pdf`)
- per-client class distributions (`<dataset>_client_split.pdf`)
- latent feature maps at the first and last round (`bloodmnist_latent.pdf`)
- McNemar test results against each baseline, and the ranges of hyperparameters the agent actually used (in `SUMMARY.txt`)

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{FedPSPO2026,
  title     = {FedPSPO: Federated Proximal Smoothing Policy Optimization for Medical Image Classification},
  author    = {Author names},
  booktitle = {Proceedings of the International Conference on Recent Challenges in Computational Intelligence and Networks (ICRCICN)},
  year      = {2026}
}
```

---

## Future work

- Combining FedPSPO with drift correction such as SCAFFOLD control variates
- Larger numbers of clients and partial participation
- Real multi-institutional data instead of simulated splits
- Differential privacy and secure aggregation
- Personalized and asynchronous federated learning

---

## License

This repository is released under the MIT License.

---

## Acknowledgements

- [MedMNIST v2](https://medmnist.com/) (Yang et al., 2023)
- [PyTorch](https://pytorch.org/)
- PPO (Schulman et al., 2017)
- FedAvg (McMahan et al., 2017), FedProx (Li et al., 2020), SCAFFOLD (Karimireddy et al., 2020)

---

## Contact

For questions or collaboration, please open an issue in this repository.
