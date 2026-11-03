# AI Engineering Masterclass

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DL-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An executable, notebook-first curriculum covering the mathematical foundations
of machine learning, classical regression and classification, data quality,
neural networks, cross-validation, regularization, and distribution-shift
analysis.

The repository is designed for progressive learning: concepts are introduced
mathematically, implemented with Python, and validated with visualizations and
evaluation metrics on synthetic and public datasets.

> **Scope:** This is an educational experimentation repository. It does not
> currently provide a production API, model registry, CI pipeline, Dockerfile,
> or a single deployed model benchmark.

## Project Overview

The notebooks demonstrate an end-to-end machine learning workflow:

1. Understand linear algebra, statistics, probability, and correlation.
2. Load and explore real datasets such as California Housing and Breast Cancer.
3. Clean data through scaling, outlier treatment, and missing-value imputation.
4. Train regression, classification, and neural-network models.
5. Compare optimization strategies and model architectures.
6. Tune hyperparameters with cross-validation and learning curves.
7. Diagnose multicollinearity, covariate shift, overfitting, and regularization.

The examples emphasize both **theory and implementation**: several algorithms
are built from scratch, then compared with scikit-learn or PyTorch equivalents.

## Technologies & Stack

- **Python 3.10+**
- **Jupyter Notebook / JupyterLab**
- **NumPy** for vectorized numerical computation
- **Pandas** for tabular data manipulation
- **SciPy** for statistical tests and scientific utilities
- **Matplotlib / Seaborn** for visualization
- **scikit-learn** for datasets, preprocessing, classical models, metrics, and
  cross-validation
- **PyTorch** for neural networks, optimization, and regularization exercises
- **statsmodels** for variance inflation factor (VIF) analysis

## Notebook Deep Dives

### 1. `ai_foundations_lecture1_and_2.ipynb`

#### Objective

Build the mathematical and statistical foundation required for later machine
learning notebooks.

#### Key Concepts

- Matrix multiplication, determinants, and matrix inversion
- Mean, median, trimmed mean, variance, and standard deviation
- Monte Carlo estimation and probability distributions
- Pearson and Spearman correlation
- Cosine similarity and its relationship to centered correlation
- Covariance matrices and multivariate distributions

#### Metrics & Results

| Metric / experiment | Result |
|---|---:|
| Pearson correlation example | **0.636** |
| Perfect-correlation validation example | **1.000** |

---

### 2. `Linear_and_Polynomial_Regression.ipynb`

#### Objective

Implement linear and polynomial regression with gradient descent, compare the
solution with the normal equation, and visualize underfitting and overfitting.

#### Key Concepts

- Linear and polynomial feature representations
- Mean squared error and cost functions
- Batch gradient descent
- Normal-equation closed-form optimization
- Feature standardization and learning-rate selection
- Overfitting as model complexity increases

#### Metrics & Results

| Experiment | Result |
|---|---:|
| Simple linear regression final parameters | `w = 0.9982`, `b = 0.0128` |
| Simple linear regression final cost | **0.0000** |
| Unstandardized gradient descent cost | **0.085213** |
| Standardized gradient descent cost | **0.036879** |
| Normal-equation final cost | **0.032252** |

---

### 3. `Week3_Regression_on_real_data.ipynb`

#### Objective

Apply regression techniques to the California Housing dataset and compare
from-scratch, scikit-learn, polynomial, and PyTorch implementations.

#### Key Concepts

- Exploratory data analysis and feature-target correlation
- California Housing regression
- Vectorized gradient descent from scratch
- Polynomial regression and model-complexity comparison
- PyTorch optimization with multiple features
- Normal-equation and scikit-learn baselines
- Residual and loss-surface visualization

#### Metrics & Results

| Model / experiment | MSE |
|---|---:|
| Degree-1 regression | **0.3506** |
| Degree-2 regression | **0.3475** |
| Degree-3 regression | **0.3421** |
| Degree-5 regression | **0.3420** |
| Best recorded multi-feature/PyTorch run | **0.3157** |
| `MedInc` single-feature baseline | **0.3506** |
| `AveRooms` single-feature baseline | **0.6926** |

Additional analysis found a target correlation of **0.688** for `MedInc` versus
**0.106** for `HouseAge`, illustrating why feature analysis matters before
training.

---

### 4. `week4_perceptron_logistic_DeepNNs.ipynb`

#### Objective

Study binary classification from the perceptron and Adaline algorithms through
logistic regression, random forests, and PyTorch deep neural networks.

#### Key Concepts

- Perceptron decision boundaries and binary updates
- Adaline and sum-of-squared-error optimization
- Sigmoid activation and logistic regression
- Binary cross-entropy / log loss
- Feature standardization
- Random forest classification
- PyTorch multilayer perceptrons
- Comparing two-feature and thirty-feature models

#### Metrics & Results

| Model / experiment | Training accuracy | Test accuracy | Additional metric |
|---|---:|---:|---:|
| Logistic Regression | — | **99.12%** | Log loss **0.0594**; mean CV **97.36%** |
| Random Forest | — | **96.49%** | Log loss **0.0957**; mean CV **95.82%** |
| Perceptron | **87.91%** | **85.09%** | — |
| Adaline | **88.35%** | **89.47%** | SSE **0.200451** |
| Logistic regression from scratch | **89.45%** | **87.72%** | Cross-entropy **0.256953** |
| PyTorch logistic regression | **89.45%** | **87.72%** | Loss **0.256990** |
| PyTorch MLP, 30 features | **99.78%** | **94.74%** | Loss **0.036886** |

The notebook also compares feature dimensionality: the two-feature network
achieved **87.72%** test accuracy, while the thirty-feature network achieved
**94.74%**.

---

### 5. `Week5_Part1_Data_Preprocessing.ipynb`

#### Objective

Demonstrate how preprocessing choices affect optimization stability, model
quality, and the distribution of training data.

#### Key Concepts

- Feature scaling and standardization
- Training-statistics-only preprocessing
- Outlier detection and percentile capping
- Mean, median, and KNN imputation
- Optimization instability caused by unscaled features
- Distribution comparison after imputation

#### Metrics & Results

| Experiment | Result |
|---|---:|
| Unscaled regression cost | **9.73 × 10³¹** |
| Scaled regression cost | **391.26** |
| Scaled + adjusted optimization cost | **378.81** |
| Cost with outliers | **0.5322** |
| Cost after capping outliers | **0.3965** |
| KNN imputation | Reported as closest to the original distribution in the notebook |

---

### 6. `Week5_Part2_Visualization_Multicollinearity.ipynb`

#### Objective

Use visual and statistical diagnostics to understand feature relationships and
identify multicollinearity before model fitting.

#### Key Concepts

- Histograms, box plots, and scatter-based EDA
- Correlation matrices
- Feature-target correlation
- Multicollinearity and unstable coefficient estimates
- Variance inflation factor (VIF)
- Feature removal and regularization as mitigation strategies

#### Metrics & Results

| Analysis | Result |
|---|---|
| Feature-target analysis | Correlation matrix generated for California Housing features |
| Multicollinearity analysis | VIF values computed with `statsmodels` |
| Outlier analysis | Box-plot workflow demonstrated on the Diabetes dataset |
| Published model score | Not the primary objective of this notebook |

This notebook focuses on diagnostic outputs rather than a single predictive
benchmark; VIF and correlation values are generated interactively when cells
are executed.

---

### 7. `Week5_Part3_CrossValidation.ipynb`

#### Objective

Build a reliable model-selection workflow using train/validation/test splits,
K-fold cross-validation, learning curves, and learning-rate tuning.

#### Key Concepts

- Separation of training, validation, and external test data
- Five-fold K-fold cross-validation
- Ten-fold cross-validation exercise
- Validation curves for learning-rate selection
- PyTorch MLP training and learning curves
- Final evaluation on an untouched test set

#### Metrics & Results

| Evaluation | Result |
|---|---:|
| Selected learning rate from five-fold CV | **0.01** |
| Final test accuracy | **96.49%** |
| Ten-fold CV accuracy at learning rate `0.001` | **97.14% ± 3.29%** |
| Dataset | Breast Cancer, **569 samples / 30 features** |

---

### 8. `Week5_Part4_Advanced_Topics.ipynb`

#### Objective

Investigate covariate shift, regularization, and learning-rate schedules as
practical tools for improving model robustness and generalization.

#### Key Concepts

- Covariate-shift simulation and diagnosis
- Overlapping coefficient (OVL)
- Kolmogorov-Smirnov (KS) test
- L1 and L2 regularization
- Phased regularization schedules
- Exponential learning-rate scheduling
- Comparing training and test behavior

#### Metrics & Results

| Experiment | Result |
|---|---:|
| OVL, feature 0 | **0.5134** |
| OVL, feature 1 | **0.4997** |
| Representative KS statistic | **0.4958**, `p < 0.000001` |
| Additional KS statistics | **0.5096**, **0.5100**, **0.5252**, **0.5094** |
| No-regularization accuracy | Train **98.00%**, test **86.67%** |
| L2-regularization accuracy | Train **87.86%**, test **85.67%** |
| Strong-L2 exercise accuracy | **85.67%** test |
| Learning-rate scheduler accuracy | **98.25%** test |

The very small KS-test p-values indicate statistically significant distribution
differences in the simulated covariate-shift examples.

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

No environment variables or secrets are required. Some notebooks download
public datasets through scikit-learn on first execution.

### Launch Jupyter

```bash
jupyter lab
```

Run each notebook from top to bottom in a fresh kernel. This ensures that
preprocessing objects, trained parameters, and visualizations are initialized
in the intended order.

### Execute a notebook non-interactively

```bash
jupyter nbconvert \
  --to notebook \
  --execute Week3_Regression_on_real_data.ipynb \
  --output /tmp/week3.executed.ipynb
```

The command writes execution results to a separate notebook and leaves the
source notebook unchanged. Interactive plots and exercise cells may still
require manual review.

### Optional Docker environment

The repository does not currently include a `Dockerfile`. A Jupyter scientific
Python image can be used as an isolated environment:

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD":/home/jovyan/work \
  quay.io/jupyter/scipy-notebook:latest \
  start-notebook.py --NotebookApp.token=''
```

Install the extra notebook dependencies from the container terminal:

```bash
python -m pip install torch statsmodels
```

Then open `http://localhost:8888` and select `/home/jovyan/work`.

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

## Contributing

Contributions that improve correctness, clarity, reproducibility, or teaching
value are welcome.

1. Fork the repository and create a focused feature branch.
2. Keep notebook examples self-contained and document new assumptions.
3. Run affected notebooks from a clean kernel before opening a pull request.
4. Remove generated checkpoints and temporary artifacts.
5. Describe validation steps and dependency changes in the pull request.

## License

This project is distributed under the [MIT License](LICENSE).
