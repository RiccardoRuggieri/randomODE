import numpy as np
import torch
import matplotlib.pyplot as plt

def _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion, forecast_horizon):
    global all_preds, all_trues
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            # forecast horizon
            _len = int(batch[0].shape[1] - forecast_horizon)

            coeffs = torch.cat((coeffs[:, :_len], torch.zeros_like(coeffs[:, _len:])), dim=1)

            optimizer.zero_grad()
            true = batch[0][:, :, 1].to(device)
            pred = model(coeffs, times).squeeze(-1)
            loss = criterion(pred, true[:, _len:])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 3 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Loss: {avg_loss}')

            ##
            model.eval()
            total_loss = 0
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch in test_loader:
                    coeffs = batch[1].to(device)
                    times = torch.linspace(0, 1, batch[0].shape[1]).to(device)
                    coeffs = torch.cat((coeffs[:, :_len], torch.zeros_like(coeffs[:, _len:])), dim=1)

                    true = batch[0][:, :, 1].to(device)
                    pred = model(coeffs, times).squeeze(-1)
                    # only compute the (pre-)forecasting loss
                    loss = criterion(pred, true[:, _len:])
                    total_loss += loss.item()

                    # all_preds.append(pred[:, _len:].cpu())
                    # all_trues.append(true[:, _len:].cpu())

                    # fill 1 - _len of pred with 0 to have a matched shape with true
                    pred = torch.cat((torch.zeros_like(true[:, :_len]), pred), dim=1)

                    all_preds.append(pred.cpu())
                    all_trues.append(true.cpu())

            avg_loss = total_loss / len(test_loader)
            print(f'Test Loss: {avg_loss}')

            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)

            num_samples = 1  # Number of samples to visualize
            forecast_start_idx = _len

            plt.figure(figsize=(12, 6))
            for i in range(num_samples):
                j = np.random.randint(0, len(all_trues))
                true_values = all_trues[j].numpy()
                pred_values = all_preds[j].numpy()

                time_indices = range(len(true_values))

                # Plot the true values as points
                plt.scatter(time_indices, true_values, color='r', label='True Values' if i == 0 else "")

                # Plot the predicted values as points
                plt.scatter(time_indices, pred_values, color='b', label='Predictions' if i == 0 else "")

                # Add lines connecting the true and predicted points
                plt.vlines(time_indices, true_values, pred_values, color='gray', alpha=0.5, linestyle='dotted')

                # Add a vertical line to indicate the forecasting start
                plt.axvline(forecast_start_idx, color='k', linestyle='--', label='Forecast Start' if i == 0 else "")

            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.ylim(0, 1.5)
            plt.title('Model Predictions vs True Values')
            plt.legend()
            plt.show()

    return all_preds, all_trues