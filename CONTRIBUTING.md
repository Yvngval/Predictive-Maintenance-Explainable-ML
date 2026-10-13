# Contributing to Predictive Maintenance and Explainable ML

Thanks for your interest in improving this project. This guide covers how to report a problem, set up the project locally and propose a change.

## Reporting bugs and suggesting features

Open an [issue](https://github.com/Yvngval/Predictive-Maintenance-Explainable-ML-/issues) and include:

- what you expected and what happened instead
- the steps to reproduce it, with the notebook and cell concerned
- your environment (operating system, Python version, versions of scikit-learn, XGBoost and SHAP)

Security issues are the exception: report them privately as described in [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/Yvngval/Predictive-Maintenance-Explainable-ML-.git
cd Predictive-Maintenance-Explainable-ML-
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On macOS, install OpenMP for XGBoost with `brew install libomp`. Download `ai4i2020.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) into `data/raw/`, then run `notebooks/01_eda.ipynb` before `notebooks/02_modeling.ipynb`.

## Proposing a change

1. Fork the repository and create a branch from `main`, for example `git checkout -b fix/threshold-plot`.
2. Keep each pull request focused on a single change.
3. Rerun the affected notebooks from top to bottom so that committed outputs and figures match the code.
4. Write commit messages in the [Conventional Commits](https://www.conventionalcommits.org/) style, for example `fix: correct legend in SHAP summary plot`.
5. Open a pull request that explains what changed and why.

## Code style

The notebooks are written in French, including markdown cells and figure labels. Keep new cells consistent with that, and save figures to `figures/` with a numbered name.
