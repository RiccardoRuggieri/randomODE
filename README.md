# Random Neural ODEs (rNODE)

This repository contains the implementation of **Random Neural ODEs (rNODE)**, a subclass of neural ordinary differential equations (NODEs) designed to incorporate stochasticity directly into the vector fields while remaining compatible with classical ODE solvers. The project was part of the course 'Numerical Methods for Mathematical Finance' held at University of Verona in the winter semester 2024/2025.

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

We stress the fact that this folder is meant ONLY for benchmark experiments, NOT for single run. Hence, plotting (whenever available) is ONLY available at the end of the whole test in the appropriate directory. For instance, in the case of ```experiments/experiments_forecasting.py``` the plots will be available at the end of the experiments at the following directories:
```output_{model_type}_{num_layers}_{hidden_dim}```
where 
1. model_type can be ```ode``` or ```sde```
2. num_layers is usually in the range ```[0, 4]```
3. hidden_dim is usually in the set ```{16, 32, 64, 128, 256}```

## Running Single Experiment

To run single experiments on a given model (i.e. a specific configuration), go in the folder ```Dataset/```, choose the task you are interested in (```Dataset/classification/'``` or ```Dataset/forecasting/```). Then, go in the folder ```launchers/```. There you will find the specific experiment. For instance, at ```Dataset/forecasting/launchers/stocks.py``` you will find the following main:
```
if __name__ == '__main__':
    # available criterion: 'L1', 'MSE', 'MAPE'
    # available type: 'ode', 'sde'
    main_classical_training(type='ode', hidden_dim=64, num_layers=1, criterion='L1')
```
which you can prompt as desired changing the number of hidden neurons, hidden layers, model type and loss criterion.

**ONLY FOR FORECASTING TASK**: plotting, in single run experiments, does not take into account stdev estimates as in the case of benchmarking experiments. On the other hand, some test samples will be plotted directly on screen (this is the case of forecasting tasks).




