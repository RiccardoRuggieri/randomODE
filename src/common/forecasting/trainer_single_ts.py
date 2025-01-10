import numpy as np
import torch
import matplotlib.pyplot as plt

def _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion):
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
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 5 == 0:
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
                    loss = criterion(pred, true)
                    total_loss += loss.item()

                    window = batch[0].to(device)

                    all_windows.append(window.cpu())
                    all_preds.append(pred.cpu())
                    all_trues.append(true.cpu())

            avg_loss = total_loss / len(test_loader)
            print(f'Test Loss: {avg_loss}')

            all_windows = torch.cat(all_windows, dim=0)
            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)

            num_samples = 1  # Number of samples to visualize

            plt.figure(figsize=(12, 6))

            for i in range(num_samples):
                j = np.random.randint(0, len(all_trues))
                input_window = all_windows[j].numpy()  # 20-day input window
                true_value = all_trues[j].numpy()       # True 21st-day value
                pred_value = all_preds[j].numpy()       # Predicted 21st-day value

                time_indices = range(20)

                # Plot the 20-day input window
                plt.plot(
                    time_indices,
                    input_window,
                    color='gray',
                    label='Input Window (20 days)' if i == 0 else ""
                )

                # Plot the true 21st-day prediction
                plt.scatter(
                    [20],
                    true_value,
                    color='r',
                    label='True 21st Day' if i == 0 else ""
                )

                # Plot the predicted 21st-day prediction
                plt.scatter(
                    [20],
                    pred_value,
                    color='b',
                    label='Predicted 21st Day' if i == 0 else ""
                )

            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.ylim(-3, 3)
            plt.title('Model Predictions vs True Values')
            plt.legend()
            plt.show()

    return all_preds, all_trues

