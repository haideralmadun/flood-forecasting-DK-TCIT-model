import numpy as np
import pandas as pd
from scipy.stats import genpareto
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv(r'tunxi 1981-2016_interpolated.csv')  # Replace with your dataset file path

# Extract streamflow data (assumed to be in the second column)
streamflow_data = dataset.iloc[:, 1].to_numpy()


def threshold_selection(streamflow_data, threshold):
    """
    Selects peaks above the given threshold and fits the Generalized Pareto Distribution (GPD).
    """
    # Identify excess values above the threshold
    excesses = streamflow_data[streamflow_data > threshold] - threshold
    
    if len(excesses) < 10:  # Require at least 10 excesses for meaningful GPD fitting
        print(f"Threshold {threshold:.2f}: Not enough excesses ({len(excesses)})")
        return None

    try:
        # Fit GPD to the excesses
        params = genpareto.fit(excesses)
        neg_log_likelihood = -np.sum(genpareto.logpdf(excesses, *params))
        return params, neg_log_likelihood
    except Exception as e:
        print(f"GPD fitting failed at threshold {threshold:.2f}: {e}")
        return None


def automated_threshold(streamflow_data):
    """
    Automates threshold selection for GPD fitting by minimizing the negative log-likelihood.
    """
    lower_percentile = np.percentile(streamflow_data, 25)  # 75th percentile
    upper_percentile = np.percentile(streamflow_data, 60)  # 95th percentile
    thresholds = np.linspace(lower_percentile, upper_percentile, 50)

    best_threshold = None
    best_params = None
    best_neg_log_likelihood = np.inf

    # Iterate over thresholds and select the best one
    for i, threshold in enumerate(thresholds):
        print(f"Evaluating threshold {threshold:.2f} ({i + 1}/{len(thresholds)})")
        result = threshold_selection(streamflow_data, threshold)

        if result is not None:
            params, neg_log_likelihood = result
            print(f"Threshold {threshold:.2f}: Negative Log-Likelihood = {neg_log_likelihood:.2f}")

            if neg_log_likelihood < best_neg_log_likelihood:
                best_threshold = threshold
                best_params = params
                best_neg_log_likelihood = neg_log_likelihood

    return best_threshold, best_params


# Run threshold selection
threshold, params = automated_threshold(streamflow_data)

if threshold is None or params is None:
    print("No suitable threshold found. Check the data or adjust the threshold range.")
else:
    print(f"Automatic threshold: {threshold:.2f}")
    print(f"Best-fit GPD parameters: {params}")

    # Identify four points between the threshold and max peak
    max_peak = np.max(streamflow_data)
    step = threshold
    points_between = []

    current_point = threshold
    for _ in range(4):
        current_point += step
        if current_point >= max_peak:
            points_between.append(max_peak)
            break
        points_between.append(current_point)

    print("Selected points between threshold and max peak:", points_between)

    # Plot results
    plt.figure(figsize=(10, 8))
    plt.plot(streamflow_data, label='Runoff (TunXi Data)', color='blue', alpha=0.6)

    # Mark the flood threshold and max peak
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=3, label=f'Flood Threshold = {threshold:.2f}')
    plt.axhline(y=max_peak, color='green', linestyle='--', linewidth=3, label=f'Max Peak = {max_peak:.2f}')

    # Mark the selected points
    for i, point in enumerate(points_between):
        y_offset = (i + 1) * 2  # Offset for text placement
        plt.axhline(y=point, color='black', linestyle='--', linewidth=3, alpha=0.7, label=f'Point {i + 1} = {point:.2f}')
        plt.text(len(streamflow_data) * 0.016, point + y_offset, f'{i + 1}', color='red', 
                 fontsize=8, fontweight='bold', ha='left', va='bottom')

    # Customize plot appearance
    plt.title('Flood Threshold Selected by POT Method with Four Peak Points Identified', fontsize=16, pad=10)
    plt.xlabel('Time (hours)', fontsize=16)
    plt.ylabel('Stream Discharge (m³/s)', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save and display plot
    plt.savefig('POT_Tunxi_four_points.tiff', dpi=600, bbox_inches='tight')
    plt.show()
