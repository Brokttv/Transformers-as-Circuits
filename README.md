# Transformers-as-Circuits
This repo is intended to be an icremental approach to studying the capacities of transformers via training on binary sequences representing different binary tasks from varying complexity classes inspired by complexity theory problems.

## Motivation

I came across this vignette from Tilde which discusses a couple of intresting findings by Merrill et al. demonstrating that a limited transformer can solve the Majority problem, which is outside the circuit class `AC^0` but inside `TC^0`.

In this repo we aim to verify that emperically and aswer the follwing question posed by the vignette:  **"Given a function that a transformer is capable of representing, can practical training setups (objective, optimizer, data, precision/compute) reliably recover it?"**

### Our Limited Transformer Config:
- `1 head`, `1 Layer`, `embedding size = 1`
- No `Positional Encodings (PE)`, `Feed-Forward-Netowrk (FFN)`, or `Layer Normalization (LN)`
- `Softmax` and `Hard-max` (saturated transformer) are used both during inference to log accuracies for each

### Key Findings:
- **Optimizer determines recovery**. Adam with embedding dimension d = 1 achieves perfect generalization to sequences 27× training length. SGD fails regardless of capacity — even at d = 1024 it plateaus at ~95% and never fully generalizes.

- **Gradient descent never finds the theoretical solution**. Merrill et al.'s construction uses uniform attention (Δ = 0). Softmax training converges instead to a token-identity clustering solution (Δ ≈ 1.10) that is consistent across all seeds but categorically distinct from theory.

- **Saturated training is unreliable**. Training with saturated attention — the regime used in Merrill et al.'s proofs — recovers a correct solution in only ~50% of runs. Softmax training succeeds in 10/10 runs with a consistent mechanism.
Undefined inputs corrupt learning. Balanced sequences on which MAJORITY is undefined are the sole cause of generalization failure on even-length training data. Removing them restores full performance. Class imbalance has no independent effect. 

- A positive control on OR, a total function, confirms even-length training works when no inputs are undefined.

### Plots:
Fig 1 illustrating both `1)` and `2)` claims:
<br>
  <p align="center">
  <img src="assets/sgd" width="600"/>
</p>
<br>

Fig 2 illustratiing claim `3)`:
<br>
  <p align="center">
  <img src="assets/ood-vs-even" width="600"/>
</p>
<br>

Fig 3 illustrating both `4)` and `5)` claims:
<br>
  <p align="center">
  <img src="assets/cluster5" width="600"/>
</p>
<br>

### Discussion:

**Striking observations**: 
- A non-saturated transformer always converges to a **token-clustering solution**
- **Saturated training finds multiple different solutions**: some clustering strongly, some near-uniform, some inverted, some learning nothing. The mechanism isn't consistent
- Transformers are sensitive to even-even train and test sequence lengths. It recovers a soltuion to Majority ~ as reliably as when trained on odd-odd pairs but only for smaller test sequence lengths. It also matters that even-even softmax fails **while saturated can sometimes rescue it** 
- **The gap between softmax and saturated inference accuracy** reveals that some learned weight configurations are only viable under one attention regime, suggesting the two regimes impose different constraints on what solutions are reachable.


---
## Setup
```bash
pip install -r requirements.txt
```

## Requirements
```
torch
numpy
tqdm
```
---
## Usage

### Basic run (Majority with AdamW)
```bash
python main.py
```


### Example 1: Majority with Soft Transformer


```bash
python main.py --task majority --epochs 50 --batch-size 128 --train-num-samples 16384 --train-seq-len 63 --test-num-samples 4096 --test-seq-len 63 --optim AdamW --lr 0.005 --data-seed 123 --device cuda
```

### Example 2: Majority with Saturatd Transformer
```bash
python main.py --task majority --epochs 80 --batch-size 256 --train-num-samples 8192 --train-seq-len 31 --test-num-samples 2048 --test-seq-len 31 --train-saturated --optim SGD --lr 0.1 --data-seed 7
```
### Example 3: OR-Task
```bash
python main.py --task or --epochs 30 --batch-size 64 --train-num-samples 8192 --train-seq-len 127 --test-num-samples 2048 --test-seq-len 127 --optim AdamW --lr 0.003 --data-seed 999 --device mps
```
---



## 📑 Citation

If you find this work useful, please cite it as:

```bibtex
@misc{brokttv2025vit,
  title        = {Transformers-on-Majority:Mapping Theory to Reality Revealing new Mechanistic Features in Transformers Learnability},
  author       = {Brokttv},
  year         = {2026},
  howpublished = {\url{https://github.com/Brokttv/Transformers-on-Majority}},
}


