# AI Engineering Masterclass

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DL-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A notebook-first, hands-on curriculum in applied machine learning and deep learning. It starts from the mathematical foundations (linear algebra, statistics, correlation), moves through regression and classification, and finishes with the practices that separate a classroom model from a trustworthy one: preprocessing, diagnostics, cross-validation, regularization, and distribution-shift analysis.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies & Stack](#technologies--stack)
- [Learning Path](#learning-path)
- [Notebook Deep Dives](#notebook-deep-dives)
  - [1. AI Foundations (Lectures 1 & 2)](#1-ai_foundations_lecture1_and_2ipynb)
  - [2. Linear and Polynomial Regression](#2-linear_and_polynomial_regressionipynb)
  - [3. Regression on Real Data (Week 3)](#3-week3_regression_on_real_dataipynb)
  - [4. Perceptron, Logistic Regression and Deep NNs (Week 4)](#4-week4_perceptron_logistic_deepnnsipynb)
  - [5. Data Preprocessing (Week 5, Part 1)](#5-week5_part1_data_preprocessingipynb)
  - [6. Visualization and Multicollinearity (Week 5, Part 2)](#6-week5_part2_visualization_multicollinearityipynb)
  - [7. Cross-Validation (Week 5, Part 3)](#7-week5_part3_crossvalidationipynb)
  - [8. Advanced Topics (Week 5, Part 4)](#8-week5_part4_advanced_topicsipynb)
- [How to Run](#how-to-run)
- [Repository Layout](#repository-layout)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This repository documents a complete, progressive path through applied AI engineering. Every notebook pairs **theory with implementation**: algorithms are first derived and coded from scratch in NumPy so the mechanics are visible, then compared against production-grade implementations in scikit-learn and PyTorch.

The notebooks collectively demonstrate an end-to-end machine learning workflow:

1. **Foundations**: linear algebra, descriptive statistics, probability, and correlation.
2. **Data exploration**: loading and inspecting real datasets (California Housing, Breast Cancer, Diabetes).
3. **Data quality**: feature scaling, outlier treatment, and missing-value imputation.
4. **Modeling**: regression, classification, and neural networks, including from-scratch and framework versions of the same model.
5. **Optimization**: gradient descent, the normal equation, learning-rate selection, and learning-rate schedules.
6. **Validation**: train/validation/test discipline, K-fold cross-validation, and learning curves.
7. **Robustness**: multicollinearity (VIF), covariate shift (OVL, KS test), and L1/L2 regularization.

> **Scope:** This is an educational and experimentation repository. It does not provide a production API, model registry, CI pipeline, Dockerfile, or a single deployed model benchmark. Metrics below are per-experiment results from the notebooks and are not directly comparable across notebooks (different datasets, splits, and objectives).

---

## Technologies & Stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Environment | Jupyter Notebook / JupyterLab |
| Numerical computing | NumPy (vectorized math, from-scratch models) |
| Data handling | Pandas |
| Statistics | SciPy (KS test, distributions), statsmodels (VIF) |
| Visualization | Matplotlib, Seaborn |
| Classical ML | scikit-learn (datasets, preprocessing, models, metrics, cross-validation) |
| Deep learning | PyTorch (autograd, `nn.Module` MLPs, optimizers, LR schedulers) |

**Datasets used:** California Housing (regression), Breast Cancer Wisconsin (binary classification, 569 samples, 30 features), Diabetes (outlier analysis), plus synthetic data for controlled experiments.

---

## Learning Path

```
Foundations ──► Regression ──► Real-world regression ──► Classification & NNs
                                                                │
        ┌───────────────────────────────────────────────────────┘
        ▼
Preprocessing ──► Diagnostics (EDA / VIF) ──► Cross-validation ──► Shift, regularization, schedulers
```

Notebooks are numbered in the recommended reading order.

---

## Notebook Deep Dives

### 1. `ai_foundations_lecture1_and_2.ipynb`

#### Objective

Build the mathematical and statistical toolkit that every later notebook relies on. The notebook implements core linear-algebra and statistics operations in NumPy and uses simulation to build intuition for probability and correlation, so later model behavior (e.g., why features get standardized, why correlated features are a problem) has a foundation.

#### Key Concepts

- **Linear algebra:** matrix multiplication, determinants, and matrix inversion (including when a matrix is singular and cannot be inverted).
- **Descriptive statistics:** mean, median, trimmed mean, variance, and standard deviation, and how each responds to outliers.
- **Probability and simulation:** Monte Carlo estimation and sampling from probability distributions.
- **Correlation:** Pearson (linear) vs. Spearman (rank-based) correlation and when they disagree.
- **Cosine similarity:** its relationship to Pearson correlation as the cosine similarity of mean-centered vectors.
- **Multivariate structure:** covariance matrices and multivariate distributions.

#### Metrics & Results

| Metric / experiment | Result |
|---|---:|
| Pearson correlation example | **0.636** |
| Perfect-correlation validation example | **1.000** |

The perfect-correlation case acts as a sanity check that the correlation implementation is correct (a variable correlated with a linear transform of itself must return 1.0).

#### How to Run

```bash
jupyter lab ai_foundations_lecture1_and_2.ipynb
```

Requires `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`. No dataset download needed; all data is synthetic or generated in-notebook. Run all cells top to bottom in a fresh kernel.

---

### 2. `Linear_and_Polynomial_Regression.ipynb`

#### Objective

Implement linear and polynomial regression from scratch using batch gradient descent, verify the result against the closed-form normal equation, and visualize how increasing model complexity moves a model from underfitting to overfitting.

#### Key Concepts

- **Linear and polynomial feature representations:** expanding a single input into polynomial terms to fit nonlinear relationships with a linear model.
- **Cost function:** mean squared error (MSE) as the objective being minimized.
- **Batch gradient descent:** iterative parameter updates using the gradient of the cost over the full dataset.
- **Normal equation:** the closed-form least-squares solution, used as a reference to validate gradient descent.
- **Feature standardization and learning rate:** why unscaled polynomial features slow or destabilize convergence.
- **Bias-variance behavior:** training error keeps falling as complexity grows while generalization degrades (overfitting).

#### Metrics & Results

| Experiment | Result |
|---|---:|
| Simple linear regression final parameters | `w = 0.9982`, `b = 0.0128` |
| Simple linear regression final cost | **0.0000** |
| Unstandardized gradient descent cost | **0.085213** |
| Standardized gradient descent cost | **0.036879** |
| Normal-equation final cost | **0.032252** |

**Reading the results:** the fitted `w ≈ 1` and `b ≈ 0` recover the underlying relationship almost exactly. Standardizing features cut the gradient-descent cost by more than half (0.0852 → 0.0369), moving it close to the normal-equation optimum (0.0323).

#### How to Run

```bash
jupyter lab Linear_and_Polynomial_Regression.ipynb
```

Requires `numpy`, `matplotlib`. Uses synthetic data, so there is no download step.

---

### 3. `Week3_Regression_on_real_data.ipynb`

#### Objective

Apply regression to the **California Housing** dataset and compare four implementations of the same idea: from-scratch gradient descent, the normal equation, scikit-learn, and PyTorch. The notebook also studies how polynomial degree and feature choice affect error.

#### Key Concepts

- **Exploratory data analysis:** feature-target correlation to decide which features carry signal.
- **Vectorized gradient descent** over multiple features.
- **Polynomial regression** at degrees 1, 2, 3, and 5, and the diminishing returns of added complexity.
- **PyTorch optimization:** `autograd`, optimizers, and training loops for multi-feature regression.
- **Baselines:** normal-equation and scikit-learn solutions as correctness references.
- **Diagnostics:** residual plots and loss-surface visualization.

#### Metrics & Results

| Model / experiment | MSE |
|---|---:|
| Degree-1 regression | **0.3506** |
| Degree-2 regression | **0.3475** |
| Degree-3 regression | **0.3421** |
| Degree-5 regression | **0.3420** |
| Best recorded multi-feature / PyTorch run | **0.3157** |
| `MedInc` single-feature baseline | **0.3506** |
| `AveRooms` single-feature baseline | **0.6926** |

**Feature analysis:** target correlation is **0.688** for `MedInc` vs. **0.106** for `HouseAge`, which explains why `MedInc` is a much stronger single predictor than `AveRooms` (MSE 0.3506 vs. 0.6926).

**Reading the results:** raising polynomial degree from 1 to 5 improves MSE only marginally (0.3506 → 0.3420), while using more features gives a bigger gain (best run 0.3157). For this dataset, adding informative features helps more than adding polynomial complexity to one feature.

#### How to Run

```bash
jupyter lab Week3_Regression_on_real_data.ipynb
```

Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`. California Housing is downloaded through scikit-learn on first execution (internet access needed once).

---

### 4. `week4_perceptron_logistic_DeepNNs.ipynb`

#### Objective

Trace the evolution of binary classification on the **Breast Cancer** dataset: from the perceptron and Adaline, through logistic regression and random forests, up to PyTorch deep neural networks. The notebook compares from-scratch and framework implementations and tests the effect of feature dimensionality.

#### Key Concepts

- **Perceptron:** binary threshold updates and linear decision boundaries.
- **Adaline:** continuous activation with sum-of-squared-error (SSE) minimization via gradient descent.
- **Logistic regression:** sigmoid activation and probabilistic outputs.
- **Binary cross-entropy (log loss):** the loss that pairs naturally with sigmoid outputs.
- **Feature standardization** for stable optimization.
- **Random forest:** an ensemble tree baseline for comparison.
- **PyTorch MLPs:** multilayer perceptrons with hidden layers and nonlinear activations.
- **Feature dimensionality:** two-feature vs. thirty-feature models.

#### Metrics & Results

| Model / experiment | Training accuracy | Test accuracy | Additional metric |
|---|---:|---:|---|
| Logistic Regression (scikit-learn) | n/a | **99.12%** | Log loss **0.0594**; mean CV **97.36%** |
| Random Forest | n/a | **96.49%** | Log loss **0.0957**; mean CV **95.82%** |
| Perceptron | **87.91%** | **85.09%** | n/a |
| Adaline | **88.35%** | **89.47%** | SSE **0.200451** |
| Logistic regression from scratch | **89.45%** | **87.72%** | Cross-entropy **0.256953** |
| PyTorch logistic regression | **89.45%** | **87.72%** | Loss **0.256990** |
| PyTorch MLP, 30 features | **99.78%** | **94.74%** | Loss **0.036886** |

**Feature dimensionality:** the two-feature network reached **87.72%** test accuracy, while the thirty-feature network reached **94.74%**.

**Reading the results:**
- The from-scratch and PyTorch logistic regressions match almost exactly (loss 0.256953 vs. 0.256990), validating both implementations.
- The scikit-learn logistic regression scores far higher than the two-feature models because it uses all 30 features.
- The 30-feature MLP shows a train/test gap (99.78% vs. 94.74%), a sign of mild overfitting that motivates the cross-validation and regularization notebooks that follow.

#### How to Run

```bash
jupyter lab week4_perceptron_logistic_DeepNNs.ipynb
```

Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`. The Breast Cancer dataset ships with scikit-learn (no download). A GPU is not required.

---

### 5. `Week5_Part1_Data_Preprocessing.ipynb`

#### Objective

Show, with measurable numbers, how preprocessing decisions change optimization stability, model quality, and the distribution of the training data.

#### Key Concepts

- **Feature scaling and standardization:** why gradient descent diverges when features sit on very different scales.
- **Training-statistics-only preprocessing:** fit scalers and imputers on the training split only, then apply them to validation/test data, to avoid data leakage.
- **Outlier detection and percentile capping (winsorization).**
- **Missing-value imputation:** mean, median, and KNN imputation.
- **Distribution comparison:** checking how each imputation method alters the original distribution.

#### Metrics & Results

| Experiment | Result |
|---|---:|
| Unscaled regression cost | **9.73 × 10³¹** |
| Scaled regression cost | **391.26** |
| Scaled + adjusted optimization cost | **378.81** |
| Cost with outliers | **0.5322** |
| Cost after capping outliers | **0.3965** |
| Best imputation method | KNN imputation (closest to the original distribution) |

**Reading the results:** the unscaled run's cost of ~10³¹ is numerical divergence, not a poor fit; scaling alone brings it to 391.26. Capping outliers reduces cost from 0.5322 to 0.3965 (about a 25% reduction). KNN imputation preserves the original distribution best because it uses relationships between features rather than a single column statistic.

#### How to Run

```bash
jupyter lab Week5_Part1_Data_Preprocessing.ipynb
```

Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

---

### 6. `Week5_Part2_Visualization_Multicollinearity.ipynb`

#### Objective

Use visual and statistical diagnostics to understand feature relationships and detect multicollinearity before fitting a model.

#### Key Concepts

- **Exploratory visualization:** histograms, box plots, and scatter plots.
- **Correlation matrices:** feature-feature and feature-target relationships.
- **Multicollinearity:** why highly correlated features inflate coefficient variance and make interpretation unreliable.
- **Variance Inflation Factor (VIF):** quantifying collinearity per feature with `statsmodels`.
- **Mitigation:** removing redundant features or applying regularization.

#### Metrics & Results

| Analysis | Output |
|---|---|
| Feature-target analysis | Correlation matrix on California Housing features |
| Multicollinearity analysis | VIF per feature, computed with `statsmodels` |
| Outlier analysis | Box-plot workflow on the Diabetes dataset |

This notebook is diagnostic rather than predictive: it has no single accuracy or MSE score. Correlation and VIF values are produced when the cells are executed. A common rule of thumb treats VIF above roughly 5 to 10 as a sign of problematic collinearity.

#### How to Run

```bash
jupyter lab Week5_Part2_Visualization_Multicollinearity.ipynb
```

Requires `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`. Datasets load through scikit-learn.

---

### 7. `Week5_Part3_CrossValidation.ipynb`

#### Objective

Build a reliable model-selection workflow: separate train, validation, and test data; tune hyperparameters with K-fold cross-validation; and read learning curves to diagnose training behavior.

#### Key Concepts

- **Data splitting discipline:** training, validation, and an untouched external test set.
- **K-fold cross-validation:** five-fold for hyperparameter search, plus a ten-fold exercise.
- **Validation curves:** choosing a learning rate by comparing cross-validated performance.
- **Learning curves** for a PyTorch MLP.
- **Final evaluation** on held-out test data, used only once after tuning.

#### Metrics & Results

| Evaluation | Result |
|---|---:|
| Dataset | Breast Cancer, **569 samples / 30 features** |
| Learning rate selected by five-fold CV | **0.01** |
| Final test accuracy | **96.49%** |
| Ten-fold CV accuracy at learning rate `0.001` | **97.14% ± 3.29%** |

**Reading the results:** the ±3.29% spread across folds shows how much a single train/test split can vary on a dataset this small, which is why a single split can be misleading. The 96.49% final test accuracy is the unbiased estimate because the test set was never used during tuning.

#### How to Run

```bash
jupyter lab Week5_Part3_CrossValidation.ipynb
```

Requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`. Cross-validation over several learning rates trains many networks, so expect longer runtime than earlier notebooks.

---

### 8. `Week5_Part4_Advanced_Topics.ipynb`

#### Objective

Investigate three practical robustness tools: covariate-shift detection, regularization, and learning-rate schedules, and measure how each affects generalization.

#### Key Concepts

- **Covariate shift:** training and test inputs drawn from different distributions, simulated and then diagnosed.
- **Overlapping coefficient (OVL):** the fraction of overlap between two distributions.
- **Kolmogorov-Smirnov (KS) test:** a nonparametric test for whether two samples come from the same distribution.
- **L1 and L2 regularization:** penalizing weight magnitude to reduce overfitting.
- **Phased regularization schedules.**
- **Exponential learning-rate scheduling.**
- **Train vs. test comparison** as the signal for overfitting.

#### Metrics & Results

**Covariate-shift diagnostics**

| Metric | Result |
|---|---:|
| OVL, feature 0 | **0.5134** |
| OVL, feature 1 | **0.4997** |
| Representative KS statistic | **0.4958** (`p < 0.000001`) |
| Additional KS statistics | **0.5096**, **0.5100**, **0.5252**, **0.5094** |

**Regularization and scheduling**

| Experiment | Train accuracy | Test accuracy |
|---|---:|---:|
| No regularization | **98.00%** | **86.67%** |
| L2 regularization | **87.86%** | **85.67%** |
| Strong-L2 exercise | n/a | **85.67%** |
| Learning-rate scheduler | n/a | **98.25%** |

**Reading the results:**
- OVL near 0.5 means the simulated train and test distributions overlap by only about half, and the KS p-values below 0.000001 confirm the difference is statistically significant.
- Without regularization the model shows a large train/test gap (98.00% vs. 86.67%). L2 narrows the gap to about 2 points (87.86% vs. 85.67%) but does not raise test accuracy here, because the gap is driven by distribution shift rather than overfitting alone.
- The exponential learning-rate schedule achieved the best test accuracy in the notebook (98.25%).

#### How to Run

```bash
jupyter lab Week5_Part4_Advanced_Topics.ipynb
```

Requires `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `torch`. Uses simulated shifted data plus scikit-learn datasets.

---

## How to Run

### Prerequisites

- Python 3.10 or newer
- `pip`
- Optional: Docker Desktop or Docker Engine

### Local setup

```bash
git clone https://github.com/AbdallahAhmed149/AI-Engineering-Masterclass.git
cd AI-Engineering-Masterclass

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install \
  jupyterlab numpy pandas scipy matplotlib seaborn \
  scikit-learn statsmodels torch
```

No environment variables or secrets are required. Some notebooks download public datasets through scikit-learn on first execution.

### Launch Jupyter

```bash
jupyter lab
```

Run each notebook from top to bottom in a fresh kernel so preprocessing objects, trained parameters, and plots are initialized in the intended order.

### Execute a notebook non-interactively

```bash
jupyter nbconvert \
  --to notebook \
  --execute Week3_Regression_on_real_data.ipynb \
  --output /tmp/week3.executed.ipynb
```

This writes results to a separate notebook and leaves the source unchanged. Interactive plots and exercise cells may still need manual review.

### Reproducibility notes

- Exact numbers can shift slightly between runs if a notebook does not fix random seeds (train/test splits, weight initialization, simulated data). Set `numpy`, `random`, and `torch` seeds for exact reproduction.
- PyTorch results can differ slightly across CPU and GPU and across library versions.

### Optional Docker environment

The repository does not include a `Dockerfile`. A Jupyter scientific Python image works as an isolated environment:

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD":/home/jovyan/work \
  quay.io/jupyter/scipy-notebook:latest \
  start-notebook.py --NotebookApp.token=''
```

Install the extra dependencies from the container terminal:

```bash
python -m pip install torch statsmodels
```

Then open `http://localhost:8888` and select `/home/jovyan/work`.

---

## Repository Layout

| Notebook | Focus |
|---|---|
| `ai_foundations_lecture1_and_2.ipynb` | Linear algebra, statistics, probability, and correlation |
| `Linear_and_Polynomial_Regression.ipynb` | Regression, gradient descent, normal equation, and overfitting |
| `Week3_Regression_on_real_data.ipynb` | California Housing regression |
| `week4_perceptron_logistic_DeepNNs.ipynb` | Perceptron, Adaline, logistic regression, random forest, and MLPs |
| `Week5_Part1_Data_Preprocessing.ipynb` | Scaling, outliers, and imputation |
| `Week5_Part2_Visualization_Multicollinearity.ipynb` | EDA, correlation, and VIF |
| `Week5_Part3_CrossValidation.ipynb` | Cross-validation and learning-rate tuning |
| `Week5_Part4_Advanced_Topics.ipynb` | Covariate shift, regularization, and schedulers |

---

## Contributing

Contributions that improve correctness, clarity, reproducibility, or teaching value are welcome.

1. Fork the repository and create a focused feature branch.
2. Keep notebook examples self-contained and document new assumptions.
3. Run affected notebooks from a clean kernel before opening a pull request.
4. Remove generated checkpoints and temporary artifacts.
5. Describe validation steps and dependency changes in the pull request.

---

## License

This project is distributed under the [MIT License](LICENSE).
