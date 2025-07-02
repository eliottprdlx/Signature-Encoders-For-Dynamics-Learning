# Experiments from "Signature-based Encoder for Dynamics Learning"

This repository provides instructions to generate all datasets used in our paper, along with the necessary files to train each model and perform the numerical experiments. Most of the code is adapted from the [Official Neural Laplace GitHub Repository](https://github.com/samholt/NeuralLaplace).

The `exp_all_baselines.py` file returns a pandas DataFrame of the test RMSE extrapolation error with standard deviation across input seed runs. This information is printed to the console and logged in a log file at the end of all seed runs. Additionally, it saves training metadata in a local `./results` folder, including the training loss array.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Repository Structure

- `./baseline_models/`: Contains all models used in the experiments.

## Running the Numerical Experiments

### Comparing Baselines

To compare all baselines across all datasets, execute the following script:

```bash
./run_all_experiments.sh
```

### Sensitivity Analysis and Ablation Study

To perform the sensitivity analysis and ablation study, run:

```bash
./run_ablation_sensitivity.sh
```

### Coupling Factor Experiment

To perform the coupling factor experiment, run:

```bash
./run_exp_coupling.sh
```

### Additional Experiments

To perform the additional experiments, run:

```bash
./run_other_tests.sh
```
