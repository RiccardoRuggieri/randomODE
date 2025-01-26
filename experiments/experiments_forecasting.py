import importlib
import pandas as pd
from common.forecasting.trainer import plot

def main(hidden_dims, num_layers, models):
    # Define configurations
    configurations = {
        "Dataset/forecasting/launchers/stocks": "Stocks",
        "Dataset/forecasting/launchers/currencies": "Currencies",
    }

    num_layers_range = range(1, num_layers) # 1, 2, 3, 4
    hidden_dims = hidden_dims # 16, 32, 64, 128

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
                        result = None
                        avg_error = 0
                        pred = None
                        predictions = []
                        for i in range(10):
                            result = experiment_module.main_classical_training(type=model_type, hidden_dim=hidden_dim, num_layers=num_layers)
                            if avg_error == 0:
                                avg_error = result["avg_error"]
                            else:
                                avg_error += result["avg_error"]
                            if pred is None:
                                pred = result["chosen_pred"]
                                predictions.append(result["chosen_pred"])
                            else:
                                pred += result["chosen_pred"]
                                predictions.append(result["chosen_pred"])

                        pred /= 10
                        avg_error /= 10

                        # plot the results of the last run shifted with stdev
                        plot(result["chosen_window"],
                             pred,
                             result["chosen_true"],
                             8,
                             result["forecast_horizon"],
                             results=result,
                             predictions=predictions)

                        result["chosen_pred"] = None
                        result["chosen_true"] = None
                        result["chosen_window"] = None

                        # Append additional metadata
                        result.update({
                            "Experiment": experiment_name,
                            "Type": model_type,
                            "Num_Layers": num_layers,
                            "Hidden_Dim": hidden_dim,
                            "AVG_AVG_ERROR": avg_error,
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
        results_path = "results_summary.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    hidden_dims = [64]
    num_layers = 2
    models = ["ode"]
    main(hidden_dims=hidden_dims, num_layers=num_layers, models=models)