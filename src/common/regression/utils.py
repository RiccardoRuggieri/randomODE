from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np

def calculate_kl_divergence(true_values, pred_values, num_bins=30):
    # Compute histogram for true and predicted values
    hist_true, bin_edges = np.histogram(true_values, bins=num_bins, density=True)
    hist_pred, _ = np.histogram(pred_values, bins=bin_edges, density=True)

    # Avoid division by zero and log(0) issues
    hist_true = np.where(hist_true == 0, 1e-10, hist_true)
    hist_pred = np.where(hist_pred == 0, 1e-10, hist_pred)

    # Calculate KL divergence
    kl_div = entropy(hist_true, hist_pred)
    return kl_div

def compare_distributions(true_data, pred_data, points, num_bins=30):
    time_points = [int(p * true_data.shape[1]) for p in points]

    fig, axes = plt.subplots(1, len(points), figsize=(20, 5), sharey=True)

    for ax, point, time_point in zip(axes, points, time_points):
        true_values = true_data[:, time_point].numpy()
        pred_values = pred_data[:, time_point].numpy()

        kl_div = calculate_kl_divergence(true_values, pred_values, num_bins)

        ax.hist([round(v,5) for v in true_values], bins=num_bins, alpha=0.5, label='True', color='r')
        ax.hist([round(v,5) for v in pred_values], bins=num_bins, alpha=0.5, label='Pred', color='b')
        ax.set_title(f'{int(point * 100)}% Point\nKL: {kl_div:.4f}')
        ax.set_xlabel('Value')
        ax.set_xlim(-0.75,1.25)
        ax.set_xticks([-0.5,0.0,0.5,1.0])
        if ax == axes[0]:
            ax.set_ylabel('Frequency')
        ax.legend()

    plt.suptitle('Distribution Comparison at Specific Points')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def show_distribution_comparison(all_trues, all_preds):
    points_to_compare = [0.2, 0.4, 0.6, 0.8]
    compare_distributions(all_trues, all_preds, points_to_compare)