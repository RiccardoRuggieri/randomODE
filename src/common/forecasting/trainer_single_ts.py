import numpy as np
import torch
import matplotlib.pyplot as plt

def _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion, forecast_horizon):
    global all_preds, all_trues
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coeffs = batch[2].to(device)
            times = torch.linspace(0, 1, batch[2].shape[1]).to(device)

            optimizer.zero_grad()
            true = batch[1].to(device).squeeze(-1).squeeze(-1)
            pred = model(coeffs, times).squeeze(-1)
            # loss only on the firs column of true
            loss = criterion(pred, true[:, :, 1].squeeze(-1))
            # loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Loss: {avg_loss}')

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
                    # batch[0][:, -1, 1]
                    loss = criterion(pred, true[:, :, 1].squeeze(-1))
                    # loss = criterion(pred, true)
                    total_loss += loss.item()

                    window = batch[0].to(device)

                    all_windows.append(window[:, :, 1].cpu())
                    # all_windows.append(window.cpu())
                    all_preds.append(pred.cpu())
                    all_trues.append(true[:, :, 1].cpu())
                    # all_trues.append(true.cpu())

            avg_loss = total_loss / len(test_loader)
            print(f'Test Loss: {avg_loss}')

            all_windows = torch.cat(all_windows, dim=0)
            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)

            plot(all_windows, all_preds, all_trues, num_samples=1, forecast_horizon=forecast_horizon)

    return all_preds, all_trues

def plot(all_windows, all_preds, all_trues, num_samples, forecast_horizon):

    plt.figure(figsize=(12, 6))

    for i in range(num_samples):
        j = np.random.randint(0, len(all_trues))
        input_window = all_windows[j].numpy()  # 20-day input window
        true_value = all_trues[j].numpy()      # True 21st-day value
        pred_value = all_preds[j].numpy()      # Predicted 21st-day value

        time_indices = range(len(input_window))

        # Plot the 20-day input window
        plt.plot(
            time_indices,
            input_window,
            color='gray',
            label='Input Window' if i == 0 else ""
        )

        # Check if true_value and pred_value are arrays
        if true_value.ndim == 1 and pred_value.ndim == 1:
            for l, (true, pred) in enumerate(zip(true_value, pred_value), start=len(input_window)):
                # Plot each true value and prediction
                plt.scatter(
                    l,
                    true,
                    color='r',
                    label='True Values' if l == len(input_window) else ""
                )
                plt.scatter(
                    l,
                    pred,
                    color='b',
                    label='Predictions' if l == len(input_window) else ""
                )
        else:
            # Handle scalar true_value and pred_value
            for l in range(len(input_window), len(input_window) + forecast_horizon):
                plt.scatter(
                    l,
                    true_value,
                    color='r',
                    label='True Values' if i == 0 else ""
                )
                plt.scatter(
                    l,
                    pred_value,
                    color='b',
                    label='Predictions' if i == 0 else ""
                )

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.ylim(-3, 3)
    plt.title('Model Predictions vs True Values')
    plt.legend()
    plt.show()
