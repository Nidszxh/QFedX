# Privacy-Preserving Quantum Federated Learning with Variational Circuits — An HPC-Based Simulation Framework

### Problem Statement:
Build a scalable simulation framework that trains Variational Quantum Circuits (VQCs) in a Federated Learning (FL) setup with basic client privacy. Use HPC to emulate multiple clients, different qubit counts, and simple noise models, and evaluate performance (wall-clock time, communication, GPU usage) and privacy–utility tradeoffs. The project compares centralized training, QFL without privacy, and QFL with basic DP + secure aggregation across datasets, qubit counts, and noise settings.


Step-by-Step Procedure (Phased plan, 10–12 weeks)

Each Phase describes goals, technical subtasks, verification checks.

## Phase 1 — Local prototype & VQC fundamentals (Weeks 1–2)

**Goal:** Implement and validate a single-client VQC training loop (no FL) on toy datasets; validate encodings, parameter-shift gradients, and readout→loss mapping.

**Tasks**
- Setup dev environment: Python 3.10+, PyTorch, PennyLane (or Qiskit/Pennylane mix), Hydra for config, MLflow for tracking. Create container (Dockerfile / Singularity recipe).
- Implement data pipeline:
  - MNIST with PCA to k features.
  - `preprocess.py`: standardize, PCA, save transformer.
- Implement VQC module `vqc.py`:
  - FeatureMap (Angle encoding) + hardware-efficient ansatz (single-qubit rotations + entanglers).
  - Readout expectation → logits → cross-entropy loss.
  - Support parameter-shift gradient and (if backend supports) adjoint differentiation.
- Train locally (single client) and verify learning on Iris and PCA-MNIST (4 qubits).

**Verification checks:**
- Loss decreases, accuracy > baseline random; gradients from parameter-shift match adjoint within tolerance.

## Phase 2 — Federated loop prototype (Weeks 3–4)

**Goal:** Build synchronous FL loop (server + multiple simulated clients as local processes) that exchanges θ and aggregates updates.

**Tasks**
- Implement FL primitives:
  - `server.py`: initialization, client sampling, aggregation (FedAvg).
  - `client.py`: receives θ, runs local E epochs, returns Δθ.
- Parameter handling: use angle deltas with wrapping to respect periodicity (wrap to [−π,π]).
- Local optimization: Adam + option for SPSA for high-cost gradient reduction.
- Simulate N clients (e.g., N=10) on single machine using multiprocessing or Ray.

**Verification Checks**
- Aggregator reconstructs mean update.
- Global model improves on held-out set vs round 0.

## Phase 3 — Privacy primitives: DP + Secure Aggregation (Weeks 5–6)

**Goal:** Integrate per-client DP (clip + Gaussian noise) and additive-mask secure aggregation; implement a privacy accountant.

**Tasks**
- Implement clipping and Gaussian noise addition:
  - Clip Δθ to ℓ2 norm C, then add Gaussian noise N(0, σ^2 C^2 I).
- Implement secure aggregation (pairwise masks):
  - Each client computes PRG-based masks with peers; masked messages cancel when summed.
  - `secure_agg.py` with seed exchange API (simulate DH key exchange during registration).
- Unit test: generate small set of Δθ, apply masks, sum masked → equals sum raw Δθ within numerical tolerance.
- Track privacy:
  - Use a ready-made library (e.g., Opacus in PyTorch) to compute ε for the given sampling rate q, noise σ, and number of rounds T.
  - Log ε after each round; no need for custom accountant.

**Verification checks**
- Mask cancellation works on toy examples.
- Opacus (or equivalent) returns plausible ε for given σ, q, T, δ.

## Phase 4 — Quantum realism & error models (Week 7)

**Goal:** Add noisy simulators: depolarizing, amplitude damping, readout error, and finite-shots measurement.

**Tasks**
- Implement `NoiseConfig` with parameters for depolarizing p, amplitude γ, readout confusion matrices and shots.
- Provide toggles in Hydra config: noise: none|depolarizing|amplitude.

**Verification**
- Compare ideal vs noisy training on small VQC; ensure noise degrades performance as expected.

## Phase 5 — HPC integration & scalability (Weeks 8–9)

**Goal:** Run simulated many-client experiments on HPC using Slurm + Ray/MPI, leverage GPU simulators where possible, implement checkpointing and logging.

**Tasks**
- Containerize stack, push Singularity image to HPC.
- Slurm orchestration:
  - Submit Ray head on a head node.
  - Launch client workers as Slurm array tasks connecting to Ray head.

- Backends:
  - Statevector for <= 20 qubits; tensor-network or cuQuantum for larger circuits.
  - Use GPU backends (Lightning.gpu, cuStateVec) when available.
- Data movement:
  - Broadcast θ via Ray object store; collect updates via RPC or object store.
- Fault tolerance:
  - Checkpoint θ every K rounds; proceed if client dropouts occur.
- Logging and experiment management:
  - MLflow run per experiment; store config, checkpoints, metrics, and final artifacts.

**Verification**
- Run a small Slurm job with 10 client tasks and 1 server; verify end-to-end execution and MLflow logs.

## Phase 6 — Experiments, ablations & analysis (Week 9–10)

**Goal:** Execute the experiment grid and compare baselines; analyze privacy–utility tradeoffs and HPC performance.

**Experiment grid (representative)**
- Datasets: Iris (4q), MNIST PCA (4–8q).
- Qubits: 2, 4, 8
- Depth L: 1, 2, 3
- Client fraction p: 0.1, 0.3, 1.0
- DP σ: 0.5, 1.0, 2.0; clip C: 0.1, 1.0
- Noise: ideal, depolarizing p=1e−3, amplitude γ=1e−3
- Baselines: centralized VQC; classical FL (TinyCNN) under same DP.

**Key metrics**
- Test accuracy ± std, AUC.
- Privacy ε (for δ set e.g. 1e−5).
- Wall-clock time & GPU hours.
- Communication per round (MB) and total.
- Robustness to non-IID (Dirichlet α∈{0.1,0.3,1.0}).

**Reporting**
- Tables: mean±std across 3–5 seeds.
- Plots: accuracy vs ε, accuracy vs qubit count, speedup vs N_clients.
- Provide code + configs + scripts to reproduce top 3 tables/plots.

## Detailed Technical Notes & Code Snippets (for submission)

**VQC forward / measurement (concept)**
- Feature mapping: RY(α * x_i) per feature.
- Ansatz block: apply RX(θ_j), RZ(φ_j) across qubits, then entangler (CNOTs).
- Readout: measure ⟨Z⟩ on readout qubit(s) → map to logit `logit = a * ⟨Z⟩ + b`.

**Parameter-shift gradient**
For parameter θ:
```
d⟨O⟩/dθ = 0.5 * (⟨O⟩_{θ + π/2} − ⟨O⟩_{θ − π/2})
```
- Implement vectorized evaluation or adjoint when supported.

**Secure aggregation (pairwise mask sketch)**
- For clients i<j: generate mask m_{i,j}. Client i adds +m_{i,j}, client j adds −m_{i,j}. Sum cancels.

**DP accountant (usage)**
- Clip + Gaussian noise, track ε with Opacus; no manual Rényi/GDP accountant required.