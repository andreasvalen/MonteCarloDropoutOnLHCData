import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def plot_histogram(values, label, threshold=0.02, original_prediction=None, std=None, show_legend=True, show_curve=True):
    # Ensure all values are between 0 and 1
    filtered_values = [v for v in values if 0 <= v <= 1]

    # Set the bin edges based on the given threshold
    bins = [i*threshold for i in range(int(1/threshold) + 2)]  # +2 to include the upper bound

    # Calculate the histogram data without plotting
    y, x, _ = plt.hist(filtered_values, bins=bins, edgecolor='black', alpha=0.7, density=True)
    plt.xlabel('Signal prediction')
    plt.ylabel('Frequency')
    plt.title(f'Histogram with {threshold} Bins')
    plt.grid(axis='y', linestyle='--')

    # Plot Gaussian distribution if show_curve is True
    if show_curve:
        mean = np.mean(filtered_values)
        std = std or np.std(filtered_values)  # Use provided std or calculate if not provided
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std)
        plt.plot(x, p, 'k', linewidth=2, label="Pred distribution")

    # Add original prediction line if provided
    if original_prediction is not None:
        plt.axvline(x=original_prediction, color='red', linestyle='--', label=f'Prediction={original_prediction:.2f}')

    # Add legend if show_legend is True
    if show_legend:
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        if label is not None:
            current_labels.append(f'Actual label = {label:.0f}')
            current_handles.append(plt.Rectangle((0, 0), 0, 0, fc="white", fill=False, edgecolor='none', linewidth=0))
        if show_curve and std is not None:
            current_labels.append(f'σ = {std:.2f}')
            current_handles.append(plt.Rectangle((0, 0), 0, 0, fc="white", fill=False, edgecolor='none', linewidth=0))
        plt.legend(handles=current_handles, labels=current_labels)

    plt.show()

def scatter_plot(preds, std, labels, degree=2):
    if len(preds) != len(std) or len(preds) != len(labels):
        print("All lists should have the same length!")
        return
    
    # Create a list of colors based on the correctness of predictions
    colors = ['blue' if ( l == 1) else 'red' for p, l in zip(preds, labels)]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, std, color=colors, alpha=0.5)
    
    # Fit a line to the data
    coefficients = np.polyfit(preds, std, degree)
    polynomial = np.poly1d(coefficients)
    
    # Generate y-values based on the fitted polynomial
    xs = np.linspace(min(preds), max(preds), 500)
    fitted_y = polynomial(xs)
    
    # Plot the fitted polynomial
    plt.plot(xs, fitted_y, color='green', label=f'Fitted Polynomial (Degree {degree})')
    
    # Create a custom legend
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Signal', markersize=10, markerfacecolor='blue')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Background', markersize=10, markerfacecolor='red')
    plt.legend(handles=[blue_patch, red_patch])
    
    plt.xlabel("Predictions")
    plt.ylabel("Standard Deviation")
    plt.title("Scatter plot of Predictions vs Standard Deviation with Fitted Line")
    plt.xlim([min(preds)-0.1, max(preds)+0.1])  # assuming predictions are between 0 and 1
    plt.ylim([0, max(std)+0.02])
    plt.show()
    
def plot_accuracy_histogram(predictions, labels, bin_size=0.05):
    if len(predictions) != len(labels):
        print("Predictions and labels must have the same length!")
        return
    
    # Grouping the predictions in bins
    bins = np.arange(0, 1 + bin_size, bin_size)
    bin_indices = np.digitize(predictions, bins) - 1
    
    correct_counts = np.zeros_like(bins[:-1])
    total_counts = np.zeros_like(bins[:-1])
    
    for pred, label in zip(predictions, labels):
        bin_index = min(int(pred / bin_size), len(bins) - 2)  # Ensure it's within range
        if (pred > 0.5 and label == 1) or (pred <= 0.5 and label == 0):
            correct_counts[bin_index] += 1
        total_counts[bin_index] += 1
    
    # Calculating accuracy for each bin
    accuracies = np.divide(correct_counts, total_counts, out=np.zeros_like(correct_counts), where=total_counts!=0) * 100
    
    # Plotting the histogram
    plt.bar(bins[:-1], accuracies, width=bin_size, alpha=0.7)
    plt.xlabel('Prediction Range')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Prediction Range')
    plt.ylim(0, 105)  # To make the y-axis range from 0 to 105 for clarity
    plt.grid(axis='y', linestyle='--')

    # Set x-ticks every 0.1 and remove bin labels
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import random

import numpy as np
import matplotlib.pyplot as plt
import random

def scatter_plot_with_4_stds(preds, std1, std2, std3, std4, labels, datapoints=10):
    if not all(len(lst) == len(preds) for lst in [std1, std2, std3, std4, labels]):
        print("All lists should have the same length!")
        return

    # Randomly select datapoints indices
    random_indices = random.sample(range(len(preds)), datapoints)

    # Colors for different groups of stds
    colors = ['blue', 'green', 'orange', 'purple']
    std_groups = [std1, std2, std3, std4]

    plt.figure(figsize=(10, 8))

    # Function to calculate bounds
    def calculate_bounds(std_group):
        upper_bounds = []
        lower_bounds = []
        bin_edges = np.arange(0, 1.05, 0.05)  # Bin edges from 0 to 1 with step of 0.05
        for bin_start in bin_edges[:-1]:
            bin_end = bin_start + 0.05
            # Filter stds in current bin
            bin_stds = [std for pred, std in zip(preds, std_group) if bin_start <= pred < bin_end]
            if bin_stds:
                sorted_stds = sorted(bin_stds)
                upper_bounds.append(sorted_stds[-10:])  # Top 10 highest values in the bin
                lower_bounds.append(sorted_stds[:10])   # Top 10 lowest values in the bin
        return np.mean(upper_bounds, axis=1), np.mean(lower_bounds, axis=1), bin_edges[:-1] + 0.025  # Middle of each bin

    # Plot each selected prediction with its respective standard deviations and calculate bounds
    for std_group, color in zip(std_groups, colors):
        upper, lower, bins = calculate_bounds(std_group)
        plt.plot(bins, upper, color=color, linestyle='-')
        plt.plot(bins, lower, color=color, linestyle='--')
        for idx in random_indices:
            plt.scatter(preds[idx], std_group[idx], color=color, alpha=0.7)

    # Create a custom legend
    std_patches = [plt.Line2D([0], [0], marker='o', color='w', label=f'STD {i+1}', markersize=10, markerfacecolor=color) 
                   for i, color in enumerate(colors)]
    plt.legend(handles=std_patches)

    plt.xlabel("Predictions")
    plt.ylabel("Standard Deviation")
    plt.title("Scatter plot of Selected Predictions vs Standard Deviations with Upper and Lower Bounds")
    plt.xlim([min(preds)-0.1, max(preds)+0.1])
    plt.ylim([0, max(max(std1), max(std2), max(std3), max(std4))+0.02])
    plt.show()




import numpy as np
import matplotlib.pyplot as plt
import random

def scatter_plot_with_dynamic_stds(preds, std_lists, labels, datapoints=10):
    # Check if all lists in std_lists and preds, labels have the same length
    if not all(len(lst) == len(preds) for lst in std_lists) or len(preds) != len(labels):
        print("All lists should have the same length!")
        return

    # Randomly select indices
    random_indices = random.sample(range(len(preds)), datapoints)

    # Generate a list of colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(std_lists)))

    plt.figure(figsize=(10, 8))

    # Plot each selected prediction with its respective standard deviations
    for idx in random_indices:
        pred = preds[idx]
        for std, color in zip([std_list[idx] for std_list in std_lists], colors):
            plt.scatter(pred, std, color=color, alpha=0.7)

    # Create a custom legend
    std_patches = [plt.Line2D([0], [0], marker='o', color='w', label=f'STD {i+1}', markersize=10, markerfacecolor=color) 
                   for i, color in enumerate(colors)]
    plt.legend(handles=std_patches)

    plt.xlabel("Predictions")
    plt.ylabel("Standard Deviation")
    plt.title("Scatter plot of Selected Predictions vs Standard Deviations")
    plt.xlim([min(preds)-0.1, max(preds)+0.1])
    plt.ylim([0, max(max(std_list) for std_list in std_lists)+0.02])
    plt.show()
