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
**1)** Optimizer matters. Adam recovers MAJORITY reliably. SGD struggles and hits a ceiling.

**2)** Embedding size matters for SGD but not Adam. Adam needs just 1 for most cases. SGD needs ~15 before it starts working at all

**3)** Training length parity matters. Train on odd, recover reliably. Train on even, softmax acc plumets in half of training seeds

**4)** A Softmax transformer reliably recovers the solution and achieves perfect accuracy under both Softmax and saturated inference

**5)** A saturated transformer is highly seed-sensitive, recovering the solution in only ~50% of runs

### Discussion:
Fig 1 illustrating both `1)` and `2)` claims:
<br>
  <p align="center">
  <img src="assets/sgd" width="950"/>
</p>
<br>

Fig 2 illustratiing claim `3)`:
<br>
  <p align="center">
  <img src="assets/ood-vs-even" width="950"/>
</p>
<br>

Fig 3 illustrating both `4)` and `5)` claims:
<br>
  <p align="center">
  <img src="assets/cluster5" width="800"/>
</p>
<br>

