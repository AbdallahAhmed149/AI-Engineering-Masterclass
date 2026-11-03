# AI Engineering Masterclass

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An executable, notebook-first curriculum for learning the foundations of machine
learning and AI engineering. The repository moves from linear algebra and
statistics through regression, classification, neural networks, data
preprocessing, cross-validation, regularization, and distribution-shift
analysis.

> **Scope:** This is a learning and experimentation repository. It does not
> currently include a production API, model registry, CI pipeline, Dockerfile,
> or a single deployed model with an application-level latency/SLA target.

## Abstract / Overview

The notebooks demonstrate how to build, inspect, train, and evaluate classical
and neural ML models from first principles and with established Python
frameworks. Examples use synthetic data and public datasets, including
California Housing, Diabetes, and Breast Cancer datasets.

The technical progression emphasizes the complete modeling lifecycle:

- Translate mathematical concepts into vectorized NumPy and PyTorch code.
- Explore data distributions, correlations, outliers, and missing values.
- Build regression and classification baselines with scikit-learn.
- Implement perceptrons, Adaline, logistic regression, and multilayer
  perceptrons with PyTorch.
- Tune learning rates with cross-validation and learning curves.
- Study regularization, learning-rate schedules, and covariate shift.

The primary value is reproducible engineering intuition: each concept is
connected to executable code, visual diagnostics, and measurable evaluation
rather than treated as an isolated algorithm.

## Architecture & Pipeline

The repository is organized as a sequence of Jupyter notebooks rather than a
packaged application:

```text
Public / synthetic dataset
          |
          v
Load -> inspect -> visualize -> clean / transform
          |
          v
Train baseline or neural model
          |
          v
Validate / tune -> evaluate on held-out data
          |
          v
Plot learning curves, decision boundaries, residuals, and diagnostics
```

### Learning workflow

1. **Foundations:** `ai_foundations_lecture1_and_2.ipynb` covers matrix
   operations, descriptive statistics, probability, and correlation.
2. **Regression:** `Linear_and_Polynomial_Regression.ipynb` and
   `Week3_Regression_on_real_data.ipynb` cover linear and polynomial regression,
   gradient descent, PyTorch optimization, and California Housing.
3. **Classification and neural networks:**
   `week4_perceptron_logistic_DeepNNs.ipynb` covers perceptrons, Adaline,
   logistic regression, random forests, and PyTorch MLPs on the Breast Cancer
   dataset.
4. **Data quality and feature analysis:** the Week 5 notebooks cover
   standardization, outlier handling, imputation, visualization, correlation,
   and variance inflation factor (VIF) analysis.
5. **Reliable evaluation and robustness:** cross-validation, learning-rate
   selection, covariate-shift detection, L1/L2 regularization, and learning-rate
   schedulers are demonstrated in the remaining Week 5 notebooks.

### Data pipeline

The notebooks use lightweight, in-notebook ETL:

1. Download or construct a public dataset.
2. Inspect schema, distributions, missing values, and outliers.
3. Split data into training, validation, and test partitions where applicable.
4. Fit preprocessing transforms on training data only, then apply them to
   validation and test data.
5. Train a baseline or neural model and compare metrics and diagnostics.

There is no external database, vector store, feature store, or persistent
inference service in the current repository.

## Performance Metrics

The following are reproducible dataset and workflow figures visible in the
notebooks. Production metrics such as request throughput and inference latency
are not claimed because this repository does not expose a serving endpoint.

| Metric | Value | Context |
|---|---:|---|
| California Housing samples | **20,640** | Regression examples |
| Breast Cancer samples | **569** | Binary classification examples |
| Breast Cancer input features | **30** | MLP and classical classification examples |
| Cross-validation configuration | **5-fold** | Learning-rate selection workflow |
| Cross-validation exercise | **10-fold** | Follow-up evaluation exercise |
| PyTorch MLP hidden width | **64 neurons** | Breast Cancer classification notebook |
| Published F1-score | **Not reported** | No repository-wide benchmark claim |
| Inference latency | **Not measured** | No production inference service |
| Inference throughput | **Not measured** | No API or load-test harness |

Model scores vary with the notebook, preprocessing choices, random seed, and
exercise configuration. Treat notebook outputs as instructional experiments,
not as a single official leaderboard.

## Tech Stack

- **Python 3.10+**
- **Jupyter Notebook / JupyterLab**
- **NumPy** for numerical computing and vectorized operations
- **Pandas** for tabular data manipulation
- **SciPy** for statistical tests and scientific utilities
- **Matplotlib** and **Seaborn** for visualization
- **scikit-learn** for datasets, preprocessing, classical models, metrics, and
  cross-validation
- **PyTorch** for optimization exercises and neural networks
- **statsmodels** for VIF and multicollinearity analysis

## Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip`
- Optional: Docker Desktop or Docker Engine for a containerized notebook
  environment

### Local installation

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

The notebooks download public datasets through scikit-learn, so an internet
connection may be required on first execution. No environment variables or
secrets are required.

### Start Jupyter

```bash
jupyter lab
```

Open the notebook that corresponds to the topic you want to study. Run cells
from top to bottom so that preprocessing, fitted parameters, and visualizations
are initialized in the expected order.

### Optional Docker workflow

The repository does not currently ship a `Dockerfile`. To run the notebooks in
an isolated Jupyter container, mount the repository into a scientific Python
image and install the project dependencies in the container:

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD":/home/jovyan/work \
  quay.io/jupyter/scipy-notebook:latest \
  start-notebook.py --NotebookApp.token=''
```

In a terminal attached to the container, install the additional dependencies
used by the notebooks:

```bash
python -m pip install torch statsmodels
```

Then open `http://localhost:8888` and select the mounted
`/home/jovyan/work` directory.

## Training & Inference

This project exposes notebook-based training rather than a command-line
training package or HTTP inference API.

### Reproduce a training workflow

```bash
jupyter nbconvert \
  --to notebook \
  --execute Week3_Regression_on_real_data.ipynb \
  --output /tmp/week3.executed.ipynb
```

For the neural-network and cross-validation workflows:

```bash
jupyter nbconvert \
  --to notebook \
  --execute Week5_Part3_CrossValidation.ipynb \
  --output /tmp/cross-validation.executed.ipynb
```

`nbconvert` executes cells in order and writes a separate output notebook,
leaving the source notebook unchanged. Interactive plots and exercises may
require manual review after execution.

### Inference

There is no standalone inference command or deployed API in the current
repository. Inference is performed inside each notebook after a model has been
trained, typically by calling a scikit-learn estimator's `predict` method or a
PyTorch model in evaluation mode:

```python
model.eval()
with torch.no_grad():
    predictions = model(features)
```

## Repository Layout

| Path | Purpose |
|---|---|
| `ai_foundations_lecture1_and_2.ipynb` | Linear algebra, statistics, probability, and correlation |
| `Linear_and_Polynomial_Regression.ipynb` | Regression concepts and overfitting |
| `Week3_Regression_on_real_data.ipynb` | California Housing regression workflow |
| `week4_perceptron_logistic_DeepNNs.ipynb` | Classical classification and PyTorch neural networks |
| `Week5_Part1_Data_Preprocessing.ipynb` | Scaling, outliers, and missing-value imputation |
| `Week5_Part2_Visualization_Multicollinearity.ipynb` | Visualization, correlation, and VIF |
| `Week5_Part3_CrossValidation.ipynb` | Cross-validation, learning curves, and hyperparameter tuning |
| `Week5_Part4_Advanced_Topics.ipynb` | Covariate shift, regularization, and schedulers |
| `LICENSE` | MIT license |

## Contributing

Contributions that improve correctness, clarity, reproducibility, or learning
value are welcome.

1. Fork the repository and create a focused feature branch.
2. Keep examples self-contained and explain new assumptions in notebook
   markdown.
3. Run affected notebooks from a clean kernel and remove generated checkpoints
   or temporary outputs before opening a pull request.
4. Describe the change, validation performed, and any dataset or dependency
   requirements in the pull request.

## License

This project is distributed under the [MIT License](LICENSE).
