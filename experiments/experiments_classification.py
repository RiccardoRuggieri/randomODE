import importlib
import pandas as pd


def main(hidden_dims, num_layers, models):
    # Define configurations
    configurations = {
        #"Dataset/classification/launchers/MIT_BIH": "MIT_BIH",
        #"Dataset/classification/launchers/sp": "SP",
        "Dataset/classification/launchers/speech_commands": "Speech_Command",
    }

    num_layers_range = range(1, num_layers) # 1, 2, 3
    hidden_dims = hidden_dims # 16, 32, 64

    # Results dictionary to store performance
    results = []

    # Iterate through configurations
    for module_path, experiment_name in configurations.items():
        module_name = module_path.replace("/", ".")
        experiment_module = importlib.import_module(module_name)

        for model_type in models:
            for num_layers in num_layers_range:
                for hidden_dim in hidden_dims:
                    print(f"Running {experiment_name} | type={model_type}, num_layers={num_layers}, hidden_dim={hidden_dim}")

                    # Capture the result (assuming main_classical_training returns a structured dictionary)
                    try:
                        result = experiment_module.main_classical_training(type=model_type, hidden_dim=hidden_dim, num_layers=num_layers)

                        # Append additional metadata
                        result.update({
                            "Experiment": experiment_name,
                            "Type": model_type,
                            "Num_Layers": num_layers,
                            "Hidden_Dim": hidden_dim,
                        })
                        results.append(result)
                    except Exception as e:
                        print(f"Error running {experiment_name} | type={model_type}, num_layers={num_layers}, hidden_dim={hidden_dim}")
                        print(e)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Handle cases where DataFrame might be empty
    if results_df.empty:
        print("No valid results were generated. Please check the main_classical_training function outputs.")
    else:
        # Save results to CSV
        results_path = "results_summary_0.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")

if __name__ == "__main__":
    hidden_dims = [32]
    num_layers = 2
    models = ["sde", "ode"]
    main(hidden_dims=hidden_dims, num_layers=num_layers, models=models)