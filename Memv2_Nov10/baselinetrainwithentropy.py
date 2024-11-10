# I need to call this file from the main.py file to train the model and get the results
import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
#import methods
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
import torch.nn.functional as F
from torch_geometric.data import Data
import sys
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from matplotlib.lines import Line2D

#planetoid_val_seeds =  [3164711608]
#planetoid_val_seeds = [3164711608,894959334,2487307261,3349051410,493067366]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")




# Global variable for save directory
def get_save_dir(dataset_name, num_epochs, num_layers, model_name, seed):
    """Create and return the save directory path with the new naming convention."""
    save_dir = Path(f'EntropyTracking/{dataset_name}_{num_epochs}epochs_{num_layers}layers_{model_name}_seed{seed}')
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def visualize_prediction_confidence_and_entropy(model, data, predictions, confidences, train_mask, test_mask, split_idx, true_labels, noise_level, save_dir):
    # Calculate KD retention (assuming this function is defined elsewhere)
    delta_entropy = kd_retention(model, data, noise_level)
    
    # Add timestamp to filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    def plot_set(mask, set_name):
        # Filter predictions for the current set (train or test)
        set_predictions = predictions[mask]
        set_confidences = confidences[mask]
        set_true_labels = true_labels[mask]
        set_delta_entropy = delta_entropy[mask]
        mask = mask.cpu()

        # Determine high delta entropy nodes (e.g., top 10%)
        high_entropy_threshold = np.percentile(set_delta_entropy, 90)
        high_entropy_mask = set_delta_entropy >= high_entropy_threshold

        # Get max confidence for each prediction
        max_confidences = torch.max(set_confidences, dim=1).values.detach().cpu().numpy()

        # Determine correctness of predictions
        correct_predictions = set_predictions == set_true_labels

        # Create a colormap for the two possible cases
        colormap = {
            True: 'green',  # Correct prediction
            False: 'red'    # Wrong prediction
        }
        colors = [colormap[cp.item()] for cp in correct_predictions]

        # Calculate category counts before creating the plot
        category_counts = {
            'green': sum(1 for c in colors if c == 'green'),
            'red': sum(1 for c in colors if c == 'red')
        }
        
        # Calculate high entropy counts for each category
        high_entropy_counts = {
            'green': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'green' and he),
            'red': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'red' and he)
        }

        # Plotting
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot low entropy nodes
        low_entropy_scatter = ax.scatter(
            max_confidences[~high_entropy_mask],
            set_delta_entropy[~high_entropy_mask],
            c=[c for c, he in zip(colors, high_entropy_mask) if not he],
            alpha=0.6,
            marker='o'
        )

        # Plot high entropy nodes with a different marker
        high_entropy_scatter = ax.scatter(
            max_confidences[high_entropy_mask],
            set_delta_entropy[high_entropy_mask],
            c=[c for c, he in zip(colors, high_entropy_mask) if he],
            alpha=0.6,
            marker='*',
            s=100  # Larger size for visibility
        )

        ax.set_title(f'Confidence vs Delta Entropy Plot ({set_name} Set, Split {split_idx})')
        ax.set_xlabel('Model Confidence')
        ax.set_ylabel('Delta Entropy')

        # Move legend outside the plot
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=f'Correct Predictions: {category_counts["green"]} (High Δ Entropy: {high_entropy_counts["green"]})', markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong Predictions: {category_counts["red"]} (High Δ Entropy: {high_entropy_counts["red"]})', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='*', color='w', label=f'High Δ Entropy: {sum(high_entropy_mask)}', markerfacecolor='gray', markersize=15),
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5))

        # Adjust layout to prevent legend overlap
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend
        
        filename_base = save_dir / f'baseline_confidence_vs_entropy_{set_name.lower()}_split_{split_idx}_{timestamp}'
        plt.savefig(f'{filename_base}.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Print the counts
        print(f"Split {split_idx} {set_name} Set Node Counts:")
        print(f"Correct Predictions: {category_counts['green']} (High Δ Entropy: {high_entropy_counts['green']})")
        print(f"Wrong Predictions: {category_counts['red']} (High Δ Entropy: {high_entropy_counts['red']})")
        print(f"Total High Δ Entropy: {sum(high_entropy_mask)}")
        print()

        # Add CSV export for training set
        if set_name == "Training":
            # Create DataFrame with the required information
            df = pd.DataFrame({
                'delta_entropy': set_delta_entropy,
                'confidence': max_confidences,
                'predicted_label': set_predictions.cpu().numpy(),
                'true_label': set_true_labels.cpu().numpy(),
                'correct_prediction': correct_predictions.cpu().numpy()
            })
            
            # Sort by delta entropy in descending order
            df = df.sort_values('delta_entropy', ascending=False)
            
            # Save to CSV
            df.to_csv(save_dir / f'baseline_training_entropy_confidence_split_{split_idx}_{timestamp}.csv', index=False)

        return category_counts, high_entropy_counts

    # Plot for test set
    test_counts, test_high_entropy_counts = plot_set(test_mask.cpu(), "Test")

    # Plot for training set
    train_counts, train_high_entropy_counts = plot_set(train_mask.cpu(), "Training")

    return {
        'test': {'counts': test_counts, 'high_entropy_counts': test_high_entropy_counts},
        'train': {'counts': train_counts, 'high_entropy_counts': train_high_entropy_counts}
    }


def kd_retention(model, data: Data, noise_level: float):
    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        out_teacher = model(data.x, data.edge_index)
        data_teacher = F.softmax(out_teacher, dim=-1).cpu().numpy()
        weight_t = np.array([entropy(dt) for dt in data_teacher])
        feats_noise = copy.deepcopy(data.x)
        feats_noise += torch.randn_like(feats_noise) * noise_level
        data_noise = Data(x=feats_noise, edge_index=data.edge_index).to(device)
    with torch.no_grad():
        out_noise = model(data_noise.x, data_noise.edge_index)
        out_noise = F.softmax(out_noise, dim=-1).cpu().numpy()
        weight_s = np.abs(np.array([entropy(on) for on in out_noise]) - weight_t)
        delta_entropy = weight_s / np.max(weight_s)
    return delta_entropy
    

def create_knn_probe(features, labels, n_neighbors=10):
    """Create and train a KNN probe."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(features, labels)
    return knn

def get_layer_representations(model, data, edge_index):
    """Get representations at each layer of the model."""
    model.eval()
    layer_representations = []
    x = data.x
    
    with torch.no_grad():
        for i, conv in enumerate(model.convs[:-1]):
            x = conv(x, edge_index)
            if hasattr(model, 'bns') and i < len(model.bns):
                x = model.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=model.dropout if hasattr(model, 'dropout') else 0.0, training=False)
            layer_representations.append(x.cpu().numpy())
        
        # Last layer
        x = model.convs[-1](x, edge_index)
        layer_representations.append(x.cpu().numpy())
    
    return layer_representations

def get_layer_names(model):
    """Get descriptive names for each layer."""
    layer_names = []
    
    # Add names for each GNN layer with their dimensions
    for i, layer in enumerate(model.convs):
        # Get input and output dimensions
        in_dim = layer.in_channels
        out_dim = layer.out_channels
        
        if i == 0:
            layer_name = f"Layer 1\n({in_dim}→{out_dim})"
        elif i == len(model.convs) - 1:
            layer_name = f"Layer {i+1}\n(Output: {in_dim}→{out_dim})"
        else:
            layer_name = f"Layer {i+1}\n({in_dim}→{out_dim})"
            
        layer_names.append(layer_name)
    
    return layer_names

def analyze_prediction_depth(model, data, train_mask, test_mask, save_dir, split_idx):
    """Analyze prediction depth using KNN probes at each layer."""
    # Get only the GNN layer representations (no input layer)
    layer_representations = get_layer_representations(model, data, data.edge_index)
    num_layers = len(layer_representations)
    
    # Calculate delta entropy for all nodes
    delta_entropy = kd_retention(model, data, noise_level=1.0)
    
    for mask_name, mask in [('train', train_mask), ('test', test_mask)]:
        print(f"\nAnalyzing {mask_name} set:")
        mask = mask.cpu().numpy()
        nodes_indices = np.where(mask)[0]
        
        # Get final GNN predictions
        model.eval()
        with torch.no_grad():
            final_out = model(data.x, data.edge_index)
            final_predictions = final_out.argmax(dim=1).cpu().numpy()
        
        node_info = {}
        
        # Calculate entropy thresholds for the current mask
        mask_entropy = delta_entropy[mask]
        high_entropy_threshold = np.percentile(mask_entropy, 75)  # top 25%
        low_entropy_threshold = np.percentile(mask_entropy, 25)   # bottom 25%
        
        for node_idx in nodes_indices:
            final_pred = final_predictions[node_idx]
            prediction_history = []
            knn_correct_history = []
            
            # Track predictions at each layer
            for layer_idx, layer_repr in enumerate(layer_representations):
                train_features = layer_repr[train_mask.cpu()]
                train_labels = data.y[train_mask].cpu()
                knn = create_knn_probe(train_features, train_labels)
                
                node_features = layer_repr[node_idx].reshape(1, -1)
                knn_pred = knn.predict(node_features)[0]
                prediction_history.append(knn_pred)
                knn_correct_history.append(knn_pred == data.y[node_idx].item())
            
            prediction_depth = num_layers - 1
            consistent_from = None
            
            if all(pred == final_pred for pred in prediction_history):
                prediction_depth = 0
                consistent_from = 0
            else:
                for i in range(len(prediction_history)):
                    if (prediction_history[i] == final_pred and 
                        all(pred == final_pred for pred in prediction_history[i:])):
                        prediction_depth = i
                        consistent_from = i
                        break
            
            # Add entropy information
            node_entropy = delta_entropy[node_idx]
            is_high_entropy = node_entropy >= high_entropy_threshold
            is_low_entropy = node_entropy <= low_entropy_threshold
            
            # Store all information for this node
            node_info[node_idx] = {
                'node_idx': node_idx,
                'prediction_depth': prediction_depth,
                'consistent_from': consistent_from,
                'prediction_history': str(prediction_history),  # Convert to string to ensure serialization
                'knn_correct_history': str(knn_correct_history),  # Convert to string to ensure serialization
                'final_prediction': final_pred,
                'true_label': data.y[node_idx].item(),
                'gnn_correct': final_pred == data.y[node_idx].item(),
                'knn_correct_at_depth': knn_correct_history[prediction_depth] if prediction_depth < len(knn_correct_history) else False,
                'delta_entropy': float(node_entropy),  # Explicitly convert to float
                'is_high_entropy': bool(is_high_entropy),
                'is_low_entropy': bool(is_low_entropy)
            }
        
        # Add debug prints before DataFrame creation
        print("Creating DataFrame with the following columns:")
        sample_node = list(node_info.values())[0]
        print("Available keys:", sample_node.keys())
        
        # Create DataFrame
        depth_df = pd.DataFrame.from_dict(node_info, orient='index')
        
        # Save the full depth analysis DataFrame
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        depth_df.to_csv(save_dir / f'full_depth_analysis_{mask_name}_split_{split_idx}_{timestamp}.csv', index=True)
        
        # Continue with visualization
        visualize_prediction_depth(depth_df, save_dir, split_idx, mask_name, model)

    return depth_df

def visualize_prediction_depth(depth_df, save_dir, split_idx, mask_name, model):
    """Create visualizations for prediction depth analysis."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Debug print at start of visualization
    print("Starting visualization with DataFrame columns:", depth_df.columns.tolist())
    
    # Verify required columns are present
    required_columns = ['prediction_depth', 'delta_entropy', 'gnn_correct']
    missing_columns = [col for col in required_columns if col not in depth_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get layer names
    layer_names = get_layer_names(model)
    depth_to_name = {i: name for i, name in enumerate(layer_names)}
    
    # 1. Distribution of Prediction Depths (existing plot)
    plt.figure(figsize=(15, 6))
    sns.histplot(data=depth_df, x='prediction_depth', bins=len(depth_df['prediction_depth'].unique()))
    plt.xticks(range(len(layer_names)), [depth_to_name[i] for i in range(len(layer_names))], rotation=45, ha='right')
    plt.title(f'Distribution of Prediction Depths ({mask_name} Set)')
    plt.xlabel('Layer')
    plt.ylabel('Number of nodes')
    plt.tight_layout()
    plt.savefig(save_dir / f'depth_distribution_{mask_name}_split_{split_idx}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Entropy vs Prediction Depth Scatter Plot
    # plt.figure(figsize=(15, 8))
    # scatter = plt.scatter(depth_df['prediction_depth'], 
    #                      depth_df['delta_entropy'],
    #                      c=depth_df['gnn_correct'], 
    #                      cmap='coolwarm',
    #                      alpha=0.6)
    
    # plt.colorbar(scatter, label='Correct Prediction')
    # plt.xticks(range(len(layer_names)), [depth_to_name[i] for i in range(len(layer_names))], rotation=45, ha='right')
    # plt.title(f'Delta Entropy vs Prediction Depth ({mask_name} Set)')
    # plt.xlabel('Prediction Depth')
    # plt.ylabel('Delta Entropy')
    # plt.tight_layout()
    # plt.savefig(save_dir / f'entropy_vs_depth_{mask_name}_split_{split_idx}_{timestamp}.png',
    #             bbox_inches='tight', dpi=300)
    # plt.close()

    # 3. Box Plot of Entropy Distribution by Prediction Depth
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='prediction_depth', y='delta_entropy', data=depth_df)
    plt.xticks(range(len(layer_names)), [depth_to_name[i] for i in range(len(layer_names))], rotation=45, ha='right')
    plt.title(f'Delta Entropy Distribution by Prediction Depth ({mask_name} Set)')
    plt.xlabel('Prediction Depth')
    plt.ylabel('Delta Entropy')
    plt.tight_layout()
    plt.savefig(save_dir / f'entropy_distribution_by_depth_{mask_name}_split_{split_idx}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # 4. Early vs Late Learners Entropy Distribution
    plt.figure(figsize=(15, 8))
    early_learners = depth_df[depth_df['prediction_depth'] <= 1]['delta_entropy']
    late_learners = depth_df[depth_df['prediction_depth'] >= 2]['delta_entropy']
    
    data_to_plot = [
        early_learners,
        late_learners
    ]
    
    violin_parts = plt.violinplot(data_to_plot, showmeans=True)
    
    # Customize violin plot colors
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    plt.xticks([1, 2], ['Early Learners\n(Depth ≤ 1)', 'Late Learners\n(Depth ≥ 2)'])
    plt.title(f'Entropy Distribution: Early vs Late Learners ({mask_name} Set)')
    plt.ylabel('Delta Entropy')
    
    # Add mean values as text
    plt.text(1, plt.ylim()[1], f'Mean: {early_learners.mean():.3f}', 
             horizontalalignment='center', verticalalignment='bottom')
    plt.text(2, plt.ylim()[1], f'Mean: {late_learners.mean():.3f}', 
             horizontalalignment='center', verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'early_vs_late_learners_entropy_{mask_name}_split_{split_idx}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # Save early vs late learners analysis
    learners_analysis = pd.DataFrame({
        'early_learners': depth_df[depth_df['prediction_depth'] <= 1]['delta_entropy'],
        'late_learners': depth_df[depth_df['prediction_depth'] >= 2]['delta_entropy']
    })
    learners_analysis.to_csv(save_dir / f'early_vs_late_learners_{mask_name}_split_{split_idx}_{timestamp}.csv')

    # 5. Heatmap of Prediction Depth vs Entropy Quantiles
    #plt.figure(figsize=(15, 8))
    
    # Create entropy quantiles
    #depth_df['entropy_quantile'] = pd.qcut(depth_df['delta_entropy'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    # Create contingency table
    #contingency = pd.crosstab(depth_df['prediction_depth'], depth_df['entropy_quantile'], normalize='all') * 100
    
    #sns.heatmap(contingency, annot=True, fmt='.1f', cmap='YlOrRd', 
    #            cbar_kws={'label': 'Percentage of Nodes'})
    
    #plt.title(f'Prediction Depth vs Entropy Quantiles ({mask_name} Set)')
    #plt.xlabel('Entropy Quantile')
    #plt.ylabel('Prediction Depth')
    #plt.tight_layout()
    #plt.savefig(save_dir / f'depth_entropy_heatmap_{mask_name}_split_{split_idx}_{timestamp}.png',
    #            bbox_inches='tight', dpi=300)
    #plt.close()

    # Continue with existing visualizations...
    # 6. KNN Probe Accuracy by Depth (existing)
    knn_accuracy_by_depth = depth_df.groupby('prediction_depth').agg({
        'node_idx': 'count',
        'knn_correct_at_depth': 'mean'
    }).rename(columns={'node_idx': 'Number of Nodes', 'knn_correct_at_depth': 'KNN Probe Accuracy'})
    
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    
    # Plot bar chart for number of nodes
    sns.barplot(x=knn_accuracy_by_depth.index, y='Number of Nodes', 
                data=knn_accuracy_by_depth, alpha=0.3, ax=ax1, color='blue')
    # Plot line for accuracy
    sns.lineplot(x=knn_accuracy_by_depth.index, y='KNN Probe Accuracy',
                data=knn_accuracy_by_depth, ax=ax2, color='red', marker='o')
    
    # Set x-axis labels to layer names
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels([depth_to_name[i] for i in range(len(layer_names))], rotation=45, ha='right')
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Nodes', color='blue')
    ax2.set_ylabel('KNN Probe Accuracy', color='red')
    plt.title(f'KNN Probe Accuracy and Node Count by Layer ({mask_name} Set)')
    plt.tight_layout()
    plt.savefig(save_dir / f'knn_depth_accuracy_{mask_name}_split_{split_idx}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # 7. GNN Model Accuracy by Depth
    gnn_accuracy_by_depth = depth_df.groupby('prediction_depth').agg({
        'node_idx': 'count',
        'gnn_correct': 'mean'
    }).rename(columns={'node_idx': 'Number of Nodes', 'gnn_correct': 'GNN Model Accuracy'})

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()

    # Plot bar chart for number of nodes
    sns.barplot(x=gnn_accuracy_by_depth.index, y='Number of Nodes',
                data=gnn_accuracy_by_depth, alpha=0.3, ax=ax1, color='blue')
    # Plot line for GNN accuracy
    sns.lineplot(x=gnn_accuracy_by_depth.index, y='GNN Model Accuracy',
                data=gnn_accuracy_by_depth, ax=ax2, color='red', marker='o')

    # Set x-axis labels to layer names
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels([depth_to_name[i] for i in range(len(layer_names))], rotation=45, ha='right')

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Nodes', color='blue')
    ax2.set_ylabel('GNN Model Accuracy', color='red')
    plt.title(f'GNN Model Accuracy and Node Count by Layer ({mask_name} Set)')
    plt.tight_layout()
    plt.savefig(save_dir / f'gnn_depth_accuracy_{mask_name}_split_{split_idx}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # 8. Create summary table
    summary_table = pd.DataFrame({
        'Total Nodes': [len(depth_df)],
        'Avg Depth': [depth_df['prediction_depth'].mean()],
        'Correct Predictions': [depth_df['gnn_correct'].sum()],
        'Early Learners (Layer 0-1)': [len(depth_df[depth_df['prediction_depth'] <= 1])],
        'Late Learners (Last Layer)': [len(depth_df[depth_df['prediction_depth'] == depth_df['prediction_depth'].max()])]
    }).T
    
    summary_table.to_csv(save_dir / f'depth_analysis_summary_{mask_name}_split_{split_idx}_{timestamp}.csv')

    # Modified GNN Accuracy vs Delta Entropy visualization
    plt.figure(figsize=(15, 8))
    
    # Create bins with approximately equal number of samples
    n_bins = 10
    # Sort delta entropy values and find bin edges that create equal-sized groups
    sorted_entropy = np.sort(depth_df['delta_entropy'])
    bin_edges = np.array_split(sorted_entropy, n_bins)
    bins = [b[0] for b in bin_edges] + [sorted_entropy[-1]]
    
    # Create the bins
    depth_df['entropy_bin'] = pd.cut(depth_df['delta_entropy'], 
                                    bins=bins,
                                    labels=range(n_bins))
    
    # Calculate accuracy and sample size for each bin
    accuracy_by_entropy = depth_df.groupby('entropy_bin').agg({
        'node_idx': 'count',
        'gnn_correct': ['mean', 'count']
    })
    
    # Flatten column names
    accuracy_by_entropy.columns = ['num_nodes', 'accuracy', 'count']
    
    # Create bin labels with sample sizes
    bin_labels = [f'{bins[i]:.3f}-{bins[i+1]:.3f}\n(n={accuracy_by_entropy.loc[i, "count"]})' 
                 for i in range(len(bins)-1)]
    
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    
    # Plot bar chart for number of nodes
    bars = sns.barplot(x=accuracy_by_entropy.index, 
                      y='num_nodes',
                      data=accuracy_by_entropy, 
                      alpha=0.3, 
                      ax=ax1,
                      color='blue')
    
    # Plot line for accuracy with error bars
    # Calculate 95% confidence intervals using binomial distribution
    confidence_intervals = []
    for idx in accuracy_by_entropy.index:
        n = accuracy_by_entropy.loc[idx, 'count']
        p = accuracy_by_entropy.loc[idx, 'accuracy']
        if n > 0:
            # Standard error for binomial distribution
            se = np.sqrt((p * (1-p)) / n)
            # 95% confidence interval
            ci = 1.96 * se
        else:
            ci = 0
        confidence_intervals.append(ci)
    
    # Plot line with error bars
    line = ax2.errorbar(x=accuracy_by_entropy.index,
                       y=accuracy_by_entropy['accuracy'],
                       yerr=confidence_intervals,
                       color='red',
                       marker='o',
                       capsize=5,
                       capthick=1,
                       elinewidth=1,
                       linestyle='-',
                       linewidth=2,
                       markersize=8)
    
    # Customize x-axis labels
    ax1.set_xticks(range(len(bin_labels)))
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    
    ax1.set_xlabel('Delta Entropy Range (sample size)')
    ax1.set_ylabel('Number of Nodes', color='blue')
    ax2.set_ylabel('GNN Model Accuracy', color='red')
    
    # Set y-axis limits for accuracy between 0 and 1
    ax2.set_ylim(-0.05, 1.05)
    
    plt.title(f'GNN Model Accuracy vs Delta Entropy ({mask_name} Set)\nwith 95% Confidence Intervals')
    
    # Add grid for easier reading
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend([(bars.patches[0]), (line)], 
              ['Node Count', 'Accuracy'],
              loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'gnn_accuracy_vs_entropy_{mask_name}_split_{split_idx}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # Save the binned analysis to CSV with proper precision
    accuracy_by_entropy['entropy_range'] = bin_labels
    accuracy_by_entropy.to_csv(save_dir / f'gnn_accuracy_vs_entropy_{mask_name}_split_{split_idx}_{timestamp}.csv')

    return accuracy_by_entropy

def train_and_get_results(data, model, optimizer, num_epochs, dataset_name, num_layers, seed, noise_level=1.0):
    """
    Train the model and track metrics.
    
    Args:
        data: Input graph data
        model: The GNN model
        optimizer: The optimizer
        num_epochs: Number of training epochs
        dataset_name: Name of the dataset
        num_layers: Number of layers in the model
        noise_level: Level of noise for entropy calculation (default: 1.0)
    """
    # Get model name from class name
    set_seed(seed)
    model_name = model.__class__.__name__
    
    # Create save_dir with all parameters including model name
    save_dir = get_save_dir(dataset_name, num_epochs, num_layers, model_name, seed)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    avg_testacc_before = []
    avg_trainacc_before = []
    avg_acc_testallsplits_before = []
    avg_acc_trainallsplits_before = []
    avg_valacc_before = []
    avg_acc_valallsplits_before = []

    def train(model, optimizer):
        model.train()
        optimizer.zero_grad()  
        out = model(data.x, data.edge_index)          
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()  
        optimizer.step()  
        pred = out.argmax(dim=1)  
        train_correct = pred[train_mask] == data.y[train_mask]  
        train_acc = int(train_correct.sum()) / int(train_mask.sum())  
        return loss

    def val(model):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_correct = pred[val_mask] == data.y[val_mask]
            val_acc = int(val_correct.sum()) / int(val_mask.sum())
        return val_acc

    def test(model):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_correct = pred[test_mask] == data.y[test_mask]
            test_acc = int(test_correct.sum()) / int(test_mask.sum())
        return test_acc, pred, out

    for split_idx in range(0, 1):
        model.reset_parameters()
        optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
        train_mask = data.train_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        
        # Check for data leakage
        train_nodes = data.train_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        test_nodes = data.test_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        val_nodes = data.val_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        
        if len(np.intersect1d(train_nodes, test_nodes)) > 0:
            print("Warning: Data leakage between train and test sets")
            sys.exit(1)
        if len(np.intersect1d(train_nodes, val_nodes)) > 0:
            print("Warning: Data leakage between train and validation sets")
            sys.exit(1)
        
        # Initialize metric storage
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        test_losses = []
        test_accs = []
        
        print(f"Training for index = {split_idx}")

        # Training loop
        for epoch in tqdm(range(1, num_epochs+1)):
            # Train step
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            # Calculate and store metrics
            with torch.no_grad():
                # Training metrics
                pred = out.argmax(dim=1)
                train_correct = pred[train_mask] == data.y[train_mask]
                train_acc = float(train_correct.sum()) / int(train_mask.sum())
                train_accs.append(train_acc * 100)
                train_losses.append(float(loss))

                # Validation metrics
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_mask], data.y[val_mask])
                val_pred = val_out.argmax(dim=1)
                val_correct = val_pred[val_mask] == data.y[val_mask]
                val_acc = float(val_correct.sum()) / int(val_mask.sum())
                val_accs.append(val_acc * 100)
                val_losses.append(float(val_loss))

                # Test metrics
                test_out = model(data.x, data.edge_index)
                test_loss = criterion(test_out[test_mask], data.y[test_mask])
                test_pred = test_out.argmax(dim=1)
                test_correct = test_pred[test_mask] == data.y[test_mask]
                test_acc = float(test_correct.sum()) / int(test_mask.sum())
                test_accs.append(test_acc * 100)
                test_losses.append(float(test_loss))

        # Plot training metrics
        plot_training_metrics(
            train_losses, train_accs,
            val_losses, val_accs,
            test_losses, test_accs,
            split_idx,
            save_dir
        )

        # Final evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            confidences = F.softmax(out, dim=1)
            
            # Calculate delta entropy here
            delta_entropy = kd_retention(model, data, noise_level)
            
            # Calculate final metrics
            train_correct = pred[train_mask] == data.y[train_mask]
            train_acc = int(train_correct.sum()) / int(train_mask.sum())
            avg_trainacc_before.append(train_acc * 100)

            val_acc = val(model)
            avg_valacc_before.append(val_acc * 100)
            
            test_acc, pred, out = test(model)
            avg_testacc_before.append(test_acc * 100)
            
            print(f'Final Training Accuracy: {train_acc*100:.4f}')
            print(f'Final Validation Accuracy: {val_acc*100:.4f}')
            print(f'Final Test Accuracy: {test_acc*100:.4f}\n')

            # # Add visualization of prediction confidence and entropy
            # visualization_results = visualize_prediction_confidence_and_entropy(
            #     model=model,
            #     data=data,
            #     predictions=pred,
            #     confidences=confidences,
            #     train_mask=train_mask,
            #     test_mask=test_mask,
            #     split_idx=split_idx,
            #     true_labels=data.y,
            #     noise_level=noise_level,
            #     save_dir=save_dir
            # )

            # Now add the prediction depth analysis
            results_df = analyze_prediction_depth(
                model=model,
                data=data,
                train_mask=train_mask,
                test_mask=test_mask,
                save_dir=save_dir,
                split_idx=split_idx
            )

        # Store average metrics
        avg_acc_trainallsplits_before.append(np.mean(avg_trainacc_before))
        avg_acc_valallsplits_before.append(np.mean(avg_valacc_before))
        avg_acc_testallsplits_before.append(np.mean(avg_testacc_before))

    return (
        avg_acc_testallsplits_before, 
        avg_acc_valallsplits_before, 
        avg_acc_trainallsplits_before,
    )

def plot_training_metrics(losses, accuracies, val_losses, val_accuracies, test_losses, test_accuracies, split_idx, save_dir):
    """Plot training, validation, and test metrics side by side."""
    epochs = range(1, len(losses) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot losses
    ax1.plot(epochs, losses, 'b-', label='Training')
    ax1.plot(epochs, val_losses, 'g-', label='Validation')
    ax1.plot(epochs, test_losses, 'r-', label='Test')
    ax1.set_title(f'Loss vs Epochs (Split {split_idx})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Set integer x-axis for epochs
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot accuracies
    ax2.plot(epochs, accuracies, 'b-', label='Training')
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation')
    ax2.plot(epochs, test_accuracies, 'r-', label='Test')
    ax2.set_title(f'Accuracy vs Epochs (Split {split_idx})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()
    
    # Set integer x-axis for epochs and appropriate y-axis for percentages
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(10))  # Set ticks every 10%
    ax2.set_ylim(-5, 105)  # Set y-axis range for percentages
    
    plt.tight_layout()
    plt.savefig(save_dir / f'baselinetraining_metrics_split_{split_idx}_{len(losses)}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': losses,
        'train_accuracy': accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'test_loss': test_losses,
        'test_accuracy': test_accuracies
    })
    metrics_df.to_csv(save_dir / f'baselinetraining_metrics_split_{split_idx}_{len(losses)}.csv', index=False)
    
    # Print final metrics
    print(f"\nFinal metrics (Split {split_idx}):")
    print(f"Training - Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.2f}%")
    print(f"Validation - Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Test - Loss: {test_losses[-1]:.4f}, Accuracy: {test_accuracies[-1]:.2f}%")















