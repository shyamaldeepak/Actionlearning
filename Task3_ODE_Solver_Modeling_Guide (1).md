# 🧪 Modeling Task 3 — ODE Solver & Sampling Step Sensitivity
### Owner: Vempadapu Shyamal Deepak
### Project: SE(3)-Equivariant Flow Matching for Sub-Grid Scale Closure in LES

---

## ✅ Why This Task Is the Easiest

| Reason | Explanation |
|---|---|
| No new model to build | You reuse the already-trained SE(3) Flow Matching model |
| No complex physics | You only change one parameter: number of ODE steps |
| Clear comparison | 3 settings → measure speed vs. quality |
| No data preprocessing | Same held-out JHTDB test cubes used by teammates |

---

## 🎯 Research Question

> **How sensitive is Flow Matching output quality to the number of ODE solver steps, and where is the optimal speed/quality trade-off?**

**Paper connection**: Hassan et al., *ET-Flow* (NeurIPS 2024) — shows Flow Matching needs far fewer sampling steps than diffusion models. Your task validates this claim in the turbulence domain.

---

## 📋 Task Summary

| Item | Details |
|---|---|
| Model | Trained SE(3) Conditional Flow Matching (from teammate Prerona) |
| Dataset | JHTDB Forced Isotropic Turbulence — held-out 64³ LES test cubes |
| Variations | 5 steps vs. 20 steps vs. 100 ODE solver steps |
| Metrics | Correlation with τᵢⱼ, sub-grid dissipation error, inference time |
| Deeper Analysis | Latency/quality trade-off curve — find the "elbow point" |

---

## 🛠️ Requirements

### Hardware / Platform
- Google Colab Free (T4 GPU) is sufficient
- Kaggle Notebooks (P100 GPU) as backup
- No local GPU needed

### Python Libraries to Install

```bash
pip install torch torchdiffeq e3nn pyJHTDB numpy scipy matplotlib wandb
```

| Library | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `torchdiffeq` | ODE solver — this is the main tool you configure |
| `e3nn` | Equivariant network (already used in model) |
| `pyJHTDB` | Access Johns Hopkins Turbulence Database |
| `matplotlib` | Plot your latency/quality curves |
| `wandb` | Log and track your experiment runs |
| `numpy / scipy` | Array math and statistics |

---

## 📚 Skills You Need to Learn

### 1. Understanding ODE Solvers (2–3 hours)
- What is an ODE solver? It numerically integrates a path from noise → sample
- In `torchdiffeq`, the key function is `odeint(func, y0, t, method='dopri5')`
- You control the number of time steps via the `t` tensor
- **Resource**: [torchdiffeq README](https://github.com/rtqichen/torchdiffeq) — read the first 2 sections

### 2. Flow Matching Basics (1–2 hours)
- Flow Matching trains a neural network to learn a velocity field
- At inference, the ODE solver follows that velocity field from t=0 to t=1
- More steps = more accurate path following, but slower
- **Resource**: Lipman et al. (2023) abstract + Section 3 only

### 3. Basic PyTorch Inference (if not already known)
- Loading a saved model: `model.load_state_dict(torch.load(...))`
- Running without gradients: `with torch.no_grad(): ...`
- Timing inference: `torch.cuda.synchronize()` before/after timing

### 4. Plotting with Matplotlib (1 hour)
- Line plots for speed vs. quality curves
- Dual-axis plots (time on one axis, correlation on the other)

---

## 📅 Step-by-Step Plan

### Week 1 — Setup & Understand
- [ ] Set up Google Colab environment, install all libraries
- [ ] Get the trained model checkpoint from teammate (Prerona/Sudip)
- [ ] Load the JHTDB held-out test cubes using `pyJHTDB`
- [ ] Run one forward pass with 20 steps to confirm everything works
- [ ] Read ET-Flow paper abstract + Section 4 (sampling efficiency)

### Week 2 — Run the 3 Experiments

#### Experiment A — 5 ODE Steps (Fast)
```python
import torch
from torchdiffeq import odeint

t_fast = torch.linspace(0, 1, 5)   # 5 steps
samples = odeint(model.velocity_field, noise, t_fast, method='dopri5')
```

#### Experiment B — 20 ODE Steps (Default)
```python
t_default = torch.linspace(0, 1, 20)   # 20 steps
samples = odeint(model.velocity_field, noise, t_default, method='dopri5')
```

#### Experiment C — 100 ODE Steps (High Quality)
```python
t_fine = torch.linspace(0, 1, 100)   # 100 steps
samples = odeint(model.velocity_field, noise, t_fine, method='dopri5')
```

For each experiment, measure:
1. **Inference time** (seconds per batch)
2. **Correlation** with true τᵢⱼ (Pearson r)
3. **Sub-grid dissipation error** (mean absolute error)

### Week 3 — Deeper Analysis: Latency/Quality Curve

Run the model at **8–10 different step counts** (not just 3):
- Steps: `[2, 5, 10, 20, 50, 100, 200, 500]`
- For each: record time + correlation
- Plot: X-axis = inference time, Y-axis = correlation with τᵢⱼ
- Identify the **elbow point** — where adding more steps stops improving quality

```python
import matplotlib.pyplot as plt

steps_list = [2, 5, 10, 20, 50, 100, 200, 500]
times = [...]       # fill from experiments
correlations = [...] # fill from experiments

fig, ax1 = plt.subplots()
ax1.plot(steps_list, correlations, 'b-o', label='Correlation')
ax1.set_xlabel('ODE Steps')
ax1.set_ylabel('Correlation with τᵢⱼ', color='b')

ax2 = ax1.twinx()
ax2.plot(steps_list, times, 'r--s', label='Inference Time (s)')
ax2.set_ylabel('Inference Time (s)', color='r')

plt.title('ODE Steps vs. Quality/Speed Trade-off')
plt.savefig('elbow_curve.png', dpi=150)
```

### Week 4 — Write Up & Finalize
- [ ] Summarize results in a table
- [ ] Write the analysis: which step count you recommend and why
- [ ] Log all runs to Weights & Biases (`wandb`)
- [ ] Prepare notebook for submission

---

## 📊 Evaluation Metrics

| Metric | How to Compute | Why It Matters |
|---|---|---|
| Pearson Correlation with τᵢⱼ | `scipy.stats.pearsonr(pred.flatten(), true.flatten())` | Main quality measure |
| Sub-grid dissipation MAE | `np.mean(np.abs(pred_diss - true_diss))` | Physics accuracy |
| Inference time (s/batch) | `time.time()` around `odeint` call | Speed measure |
| Memory usage (MB) | `torch.cuda.memory_allocated() / 1e6` | Optional, good to report |

---

## 🔍 Deeper Analysis — The Elbow Point

The key insight to find and report:

1. **At very few steps (2–5)**: Fast but inaccurate — the ODE solver cuts corners
2. **At moderate steps (20–50)**: Good balance — ET-Flow paper claims this is sufficient
3. **At many steps (100–500)**: Diminishing returns — quality plateaus but time grows linearly

**Your conclusion should state**: "X steps is the optimal trade-off for this turbulence task, achieving Y% of the quality of 500 steps at Z% of the compute cost."

---

## ⚠️ Risks & Backup Plan

| Risk | Likelihood | Backup Plan |
|---|---|---|
| Trained model not ready from teammates | Medium | Use a simpler non-equivariant Flow Matching model as stand-in |
| JHTDB access takes time to set up | Low | Use synthetic Gaussian noise data to test the pipeline first |
| Colab session times out on long runs | Medium | Save checkpoints every experiment; use Kaggle for longer runs |

---

## 📁 Deliverable Checklist

- [ ] Jupyter notebook with all 3 experiments clearly labeled
- [ ] Latency/quality elbow curve plot saved as PNG
- [ ] Results table (steps → time → correlation → dissipation error)
- [ ] 1-page PDF task summary (for submission)
- [ ] WandB run link shared with team

---

## 🔗 Key Resources

| Resource | Link |
|---|---|
| torchdiffeq (ODE solver library) | https://github.com/rtqichen/torchdiffeq |
| ET-Flow paper (NeurIPS 2024) | https://arxiv.org/abs/2410.22388 |
| Flow Matching paper (ICLR 2023) | https://arxiv.org/abs/2210.02747 |
| JHTDB Python client | https://github.com/idies/pyJHTDB |
| Weights & Biases quickstart | https://docs.wandb.ai/quickstart |
| Google Colab GPU guide | https://colab.research.google.com/notebooks/gpu.ipynb |
