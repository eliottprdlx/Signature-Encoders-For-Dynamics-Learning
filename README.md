# Experimental code for "Learning non-Markovian Dynamical Systems with Signatured-based Encoders"

This repository provides instructions to generate all datasets used in our paper, along with the necessary files to train each model and perform the numerical experiments. Most of the code is adapted from the [Neural Laplace GitHub Repository](https://github.com/samholt/NeuralLaplace). To get started with the Signatory library, we recommend consulting the [Signatory Documentation](https://signatory.readthedocs.io/en/latest/).

The `exp_all_baselines.py`, `ablation_study.py` and `sensitivity_analysis.py` file returns a pandas DataFrame of the test RMSE extrapolation error with standard deviation across input seed runs. This information is printed to the console and logged in a log file at the end of all seed runs. Additionally, it saves training metadata in a local `./results` folder, including the training loss array and the state of the best model. The `other_tests.py` file only returns a pandas DataFrame of the test RMSE extrapolation error with standard deviation across input seed runs.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Repository Structure

`./baseline_models/`: Contains all models used in the experiments. 

## Running the Numerical Experiments

### Comparing Baselines

To compare all baselines across all datasets, execute the script:

```bash
./run_baselines_comparison.sh
```

### Ablation Study

To perform the ablation study, execute the script:

```bash
./run_ablation_study.sh
```

### Sensitivity Analysis

To perform the sensitivity analysis, execute the script:

```bash
./run_sensitivity_analysis.sh
```

### Coupling Factor Experiment

To perform the coupling factor experiment, execute the script:

```bash
./run_exp_coupling.sh
```

### Additional Experiments

To perform the additional experiments, make sure to first run `run_baselines_comparison.sh`, then execute the script:

```bash
./run_other_tests.sh
```
