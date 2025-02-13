import torch

def _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion, forecast_horizon, mean, std, with_stdev=False):
    global all_preds, all_trues

    results = {
        "avg_error": None,
        "chosen_true": None,
        "chosen_window": None,
        "chosen_pred": None,
        "forecast_horizon": None,
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coeffs = batch[2].to(device)
            times = torch.linspace(0, 1, batch[2].shape[1]).to(device)

            optimizer.zero_grad()
            true = batch[1].to(device).squeeze(-1).squeeze(-1)
            pred = model(coeffs, times).squeeze(-1)
            loss = criterion(pred, true[:, :, 1].squeeze(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 1 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}')
            print(f'Train Loss: {avg_loss}')

            ##
            model.eval()
            total_loss = 0
            all_preds = []
            all_trues = []
            all_windows = []
            with torch.no_grad():
                for batch in test_loader:
                    coeffs = batch[2].to(device)
                    times = torch.linspace(0, 1, batch[2].shape[1]).to(device)

                    true = batch[1].to(device).squeeze(-1).squeeze(-1)
                    pred = model(coeffs, times).squeeze(-1)

                    # denormalize
                    pred = pred * std[1] + mean[1]
                    true = true * std[1] + mean[1]

                    loss = criterion(pred, true[:, :, 1].squeeze(-1))
                    total_loss += loss.item()

                    window = batch[0].to(device)
                    window = window * std[1] + mean[1]

                    all_windows.append(window[:, :, 1].cpu())
                    all_preds.append(pred.cpu())
                    all_trues.append(true[:, :, 1].cpu())

            avg_loss = total_loss / len(test_loader)
            print(f'Test Loss: {avg_loss}')

            all_windows = torch.cat(all_windows, dim=0)
            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)

            # Denormalize
            # all_preds = all_preds * std[1] + mean[1]
            # all_trues = all_trues * std[1] + mean[1]
            # all_windows = all_windows * std[1] + mean[1]
            if epoch % 10 == 0:
                plot_withoutStdev(all_windows, all_preds, all_trues, num_samples=1, forecast_horizon=forecast_horizon)

    if with_stdev:
        results = plot(all_windows, all_preds, all_trues, num_samples=10, forecast_horizon=forecast_horizon, results=results)
        results["avg_error"] = avg_loss
    else:
        results = plot_withoutStdev(all_windows, all_preds, all_trues, num_samples=5, forecast_horizon=forecast_horizon)

    return results

def plot(all_windows, all_preds, all_trues, num_samples, forecast_horizon, results=None, predictions=None, model_type='ode_1_32'):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # here path
    here = os.path.dirname(os.path.abspath(__file__))

    output_dir = here + '/output' + '/' + model_type

    if results is not None:
        results["chosen_pred"] = all_preds
        results["chosen_true"] = all_trues
        results["chosen_window"] = all_windows
        results["forecast_horizon"] = forecast_horizon

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        j = i  # or use np.random.randint(0, len(all_trues)) for randomness
        input_window = all_windows[j].numpy()  # 20-day input window
        true_value = all_trues[j].numpy()      # True 21st-day value
        pred_value = all_preds[j].numpy()      # Predicted 21st-day value

        stdev = 0

        if predictions is not None:
            # compute the variance of the predictions[j]
            stdev = np.std([pred[j].numpy() for pred in predictions])

        # Combine all values for the current sample to compute y-limits
        sample_values = np.concatenate([input_window.flatten(), true_value.flatten(), pred_value.flatten()])
        y_min, y_max = np.min(sample_values), np.max(sample_values)
        padding = (y_max - y_min) * 0.4  # Add 10% padding
        y_min, y_max = y_min - padding, y_max + padding

        time_indices = range(len(input_window))

        plt.figure(figsize=(12, 6), dpi=100)

        # Plot the 20-day input window as a line
        plt.plot(
            time_indices,
            input_window,
            color='gray',
            label='Input Window' if i == 0 else ""
        )

        # Plot the predicted values with the standard deviation as a shaded region
        forecast_indices = range(len(input_window), len(input_window) + forecast_horizon)

        # Plot the prediction line
        plt.plot(
            forecast_indices,
            pred_value,
            color='b',
            label='Predictions' if i == 0 else ""
        )

        # Plot the true value line
        plt.plot(
            forecast_indices,
            true_value,
            color='r',
            label='True Values' if i == 0 else ""
        )

        # Plot the standard deviation shading around the predictions
        if predictions is not None:
            plt.fill_between(
                forecast_indices,
                pred_value - stdev,
                pred_value + stdev,
                color='blue', alpha=0.2, label='Standard Dev' if i == 0 else ""
            )

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.ylim(y_min, y_max)  # Dynamically set the y-limits for the current sample

        file_path = os.path.join(output_dir, f"sample_{i+1}.png")
        plt.savefig(file_path)
        plt.close()  # Close the plot to free up memory

    return results

def plot_withoutStdev(all_windows, all_preds, all_trues, num_samples, forecast_horizon, predictions=None):
    import numpy as np
    import matplotlib.pyplot as plt

    for i in range(num_samples):
        j = i  # or use np.random.randint(0, len(all_trues)) for randomness
        input_window = all_windows[j].numpy()  # 20-day input window
        true_value = all_trues[j].numpy()      # True 21st-day value
        pred_value = all_preds[j].numpy()      # Predicted 21st-day value

        stdev = 0

        if predictions is not None:
            # compute the variance of the predictions[j]
            stdev = np.std([pred[j].numpy() for pred in predictions])

        # Combine all values for the current sample to compute y-limits
        sample_values = np.concatenate([input_window.flatten(), true_value.flatten(), pred_value.flatten()])
        y_min, y_max = np.min(sample_values), np.max(sample_values)
        padding = (y_max - y_min) * 0.4  # Add 10% padding
        y_min, y_max = y_min - padding, y_max + padding

        time_indices = range(len(input_window))

        plt.figure(figsize=(12, 6), dpi=100)

        # Plot the 20-day input window as a line
        plt.plot(
            time_indices,
            input_window,
            color='gray',
            label='Input Window' if i == 0 else ""
        )

        # Plot the predicted values with the standard deviation as a shaded region
        forecast_indices = range(len(input_window), len(input_window) + forecast_horizon)

        # Plot the prediction line
        plt.plot(
            forecast_indices,
            pred_value,
            color='b',
            label='Predictions' if i == 0 else ""
        )

        # Plot the true value line
        plt.plot(
            forecast_indices,
            true_value,
            color='r',
            label='True Values' if i == 0 else ""
        )

        # Plot the standard deviation shading around the predictions
        if predictions is not None:
            plt.fill_between(
                forecast_indices,
                pred_value - stdev,
                pred_value + stdev,
                color='blue', alpha=0.2, label='Standard Dev' if i == 0 else ""
            )

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.ylim(y_min, y_max)  # Dynamically set the y-limits for the current sample

        # Save the plot to the output directory
        plt.show()
        plt.close()  # Close the plot to free up memory




