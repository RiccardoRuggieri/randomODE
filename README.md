# Random Neural ODEs (rNODE)

This repository contains the implementation of **Random Neural ODEs (rNODE)**, a subclass of neural ordinary differential equations (NODEs) designed to incorporate stochasticity directly into the vector fields while remaining compatible with classical ODE solvers.

## Repository Structure

- **`src/`**: Core implementation of rNODEs, including vector field definitions and solvers.
- **`experiments/`**: Scripts to reproduce the experiments (mainly as ensambles) described in the project, such as time series forecasting and classification tasks.
- **`Dataset/`**: Data preprocessing of each experiment, toghether with the launchers of single experiments.

## Running Experiments

To run the experiments, go in experiments' folder and hardcode the main launcher
```
if __name__ == "__main__":
    hidden_dims = [16, 32, 64, 128]
    num_layers = 3
    main(hidden_dims=hidden_dims, num_layers=num_layers)
```
