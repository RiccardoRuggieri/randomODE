# Random Neural ODEs (rNODE)

This repository contains the implementation of **Random Neural ODEs (rNODE)**, a subclass of neural ordinary differential equations (NODEs) designed to incorporate stochasticity directly into the vector fields while remaining compatible with classical ODE solvers.

## Repository Structure

- **`src/`**: Core implementation of rNODEs, including vector field definitions and solvers.
- **`experiments/`**: Scripts to reproduce the experiments (mainly as ensambles) described in the project, such as time series forecasting and classification tasks.
- **`Dataset/`**: Data preprocessing of each experiment, toghether with the launchers of single experiments.

## Install Dependencies

Just run

```
git clone https://github.com/RiccardoRuggieri/randomODE.git
cd randomODE/src
pip install -r requirements.txt
```

## Running Experiments

To run the experiments, go in experiments' folder and hardcode the main launcher. I.e.,
```
if __name__ == "__main__":
    hidden_dims = [16, 32, 64, 128]
    num_layers = 3
    models = ["ode", "sde"]
    main(hidden_dims=hidden_dims, num_layers=num_layers, models=models)
```
where
1. **hidden_dims** is (a list of) the number of hidden neuron for each hidden layer.
2. **num_layers** is the number of layer of the vector field associated with the neural (random) ODE.
3. **models** is (a list of) models we want to test on the given experiments. Available models are ```ode```, ```sde``` where with ode we are reffering to random neural ODE, while with sde to controlled neural SDE.


