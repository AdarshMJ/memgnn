import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import seaborn as sns

def create_aggregate_training_plot(all_metrics, save_dir):
    """Create aggregate training metrics plot with confidence intervals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    epochs = range(1, len(all_metrics['train']['accuracy'][0]) + 1)
    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    
    # Plot losses
    for metric_type in ['train', 'val', 'test']:
        losses = np.array(all_metrics[metric_type]['loss'])
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        ci_loss = 1.96 * std_loss / np.sqrt(len(losses))
        
        ax1.plot(epochs, mean_loss, color=colors[metric_type], label=f'{metric_type.capitalize()}')
        ax1.fill_between(epochs, mean_loss - ci_loss, mean_loss + ci_loss, 
                        color=colors[metric_type], alpha=0.2)
    
    ax1.set_title('Loss vs Epochs (with 95% CI)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot accuracies
    for metric_type in ['train', 'val', 'test']:
        accs = np.array(all_metrics[metric_type]['accuracy'])
        mean_acc = np.mean(accs, axis=0)
        std_acc = np.std(accs, axis=0)
        ci_acc = 1.96 * std_acc / np.sqrt(len(accs))
        
        ax2.plot(epochs, mean_acc, color=colors[metric_type], label=f'{metric_type.capitalize()}')
        ax2.fill_between(epochs, mean_acc - ci_acc, mean_acc + ci_acc,
                        color=colors[metric_type], alpha=0.2)
    
    ax2.set_title('Accuracy vs Epochs (with 95% CI)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aggregate_training_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_aggregate_depth_distribution(all_depth_distributions, save_dir):
    """Create aggregate depth distribution plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for set_type, ax in zip(['train', 'test'], [ax1, ax2]):
        depth_data = [df['prediction_depth'] for seed, st, df in all_depth_distributions if st == set_type]
        
        # Calculate mean and CI for each depth
        all_depths = np.unique(np.concatenate([d.unique() for d in depth_data]))
        depth_counts = np.zeros((len(depth_data), len(all_depths)))
        
        for i, depths in enumerate(depth_data):
            counts = depths.value_counts()
            for j, depth in enumerate(all_depths):
                depth_counts[i, j] = counts.get(depth, 0)
        
        # Normalize to percentages
        depth_percentages = depth_counts / depth_counts.sum(axis=1, keepdims=True) * 100
        mean_percentages = np.mean(depth_percentages, axis=0)
        ci_percentages = 1.96 * np.std(depth_percentages, axis=0) / np.sqrt(len(depth_data))
        
        ax.bar(all_depths, mean_percentages, yerr=ci_percentages, capsize=5)
        ax.set_title(f'Prediction Depth Distribution ({set_type.capitalize()} Set)')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Percentage of Nodes')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aggregate_depth_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_aggregate_accuracy_depth_plots(all_depth_distributions, save_dir):
    """Create aggregate accuracy vs depth plots for both GNN and KNN."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    for set_type, (ax_acc, ax_knn) in zip(['train', 'test'], [(ax1, ax3), (ax2, ax4)]):
        depth_dfs = [df for seed, st, df in all_depth_distributions if st == set_type]
        
        gnn_accuracies = []
        knn_accuracies = []
        node_counts = []
        
        for df in depth_dfs:
            depth_groups = df.groupby('prediction_depth')
            gnn_acc = depth_groups['gnn_correct'].mean()
            gnn_accuracies.append(gnn_acc)
            knn_acc = depth_groups['knn_correct_at_depth'].mean()
            knn_accuracies.append(knn_acc)
            counts = depth_groups.size()
            node_counts.append(counts)
        
        gnn_accuracies = np.array(pd.concat(gnn_accuracies, axis=1))
        knn_accuracies = np.array(pd.concat(knn_accuracies, axis=1))
        node_counts = np.array(pd.concat(node_counts, axis=1))
        
        gnn_mean = np.mean(gnn_accuracies, axis=1)
        gnn_ci = 1.96 * np.std(gnn_accuracies, axis=1) / np.sqrt(len(depth_dfs))
        knn_mean = np.mean(knn_accuracies, axis=1)
        knn_ci = 1.96 * np.std(knn_accuracies, axis=1) / np.sqrt(len(depth_dfs))
        count_mean = np.mean(node_counts, axis=1)
        
        # Plot GNN accuracy
        ax_acc.bar(range(len(count_mean)), count_mean, alpha=0.3, color='blue')
        ax_acc2 = ax_acc.twinx()
        ax_acc2.errorbar(range(len(gnn_mean)), gnn_mean, yerr=gnn_ci,
                        color='red', marker='o', capsize=5)
        
        ax_acc.set_title(f'GNN Accuracy vs Depth ({set_type.capitalize()} Set)')
        ax_acc.set_xlabel('Layer')
        ax_acc.set_ylabel('Number of Nodes', color='blue')
        ax_acc2.set_ylabel('GNN Accuracy', color='red')
        ax_acc2.set_ylim(-0.05, 1.05)
        
        # Plot KNN accuracy
        ax_knn.bar(range(len(count_mean)), count_mean, alpha=0.3, color='blue')
        ax_knn2 = ax_knn.twinx()
        ax_knn2.errorbar(range(len(knn_mean)), knn_mean, yerr=knn_ci,
                        color='red', marker='o', capsize=5)
        
        ax_knn.set_title(f'KNN Probe Accuracy vs Depth ({set_type.capitalize()} Set)')
        ax_knn.set_xlabel('Layer')
        ax_knn.set_ylabel('Number of Nodes', color='blue')
        ax_knn2.set_ylabel('KNN Accuracy', color='red')
        ax_knn2.set_ylim(-0.05, 1.05)
        
        # Add legends
        for ax in [ax_acc, ax_knn]:
            ax.legend(['Node Count'], loc='upper left')
        for ax in [ax_acc2, ax_knn2]:
            ax.legend(['Accuracy'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aggregate_accuracy_depth.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_aggregate_entropy_accuracy_plot(all_entropy_accuracies, save_dir):
    """Create aggregate entropy vs accuracy plot with detailed x-axis labels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for set_type, ax in zip(['train', 'test'], [ax1, ax2]):
        entropy_dfs = all_entropy_accuracies[set_type]
        if not entropy_dfs:
            continue
            
        entropy_ranges = entropy_dfs[0]['entropy_range'].str.extract(r'([\d.]+)-([\d.]+)')[0].astype(float)
        accuracies = np.array([df['accuracy'].values for df in entropy_dfs])
        mean_acc = np.mean(accuracies, axis=0)
        ci_acc = 1.96 * np.std(accuracies, axis=0) / np.sqrt(len(accuracies))
        node_counts = np.mean([df['num_nodes'].values for df in entropy_dfs], axis=0)
        
        bars = ax.bar(range(len(mean_acc)), node_counts, alpha=0.3, color='blue')
        ax2 = ax.twinx()
        line = ax2.errorbar(range(len(mean_acc)), mean_acc, yerr=ci_acc,
                          marker='o', capsize=5, color='red')
        
        x_labels = []
        for i, (df, acc) in enumerate(zip(entropy_dfs[0].itertuples(), mean_acc)):
            entropy_range = df.entropy_range.split('\n')[0]
            avg_nodes = int(np.mean([d['num_nodes'].iloc[i] for d in entropy_dfs]))
            x_labels.append(f'{entropy_range}\n(n={avg_nodes})')
        
        ax.set_xticks(range(len(mean_acc)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlabel('Delta Entropy Range (average sample size)')
        ax.set_ylabel('Number of Nodes', color='blue')
        ax2.set_ylabel('Accuracy', color='red')
        ax2.set_ylim(-0.05, 1.05)
        ax.set_title(f'Accuracy vs Delta Entropy ({set_type.capitalize()} Set)')
        
        ax.legend([bars.patches[0]], ['Node Count'], loc='upper left')
        ax2.legend([line], ['Accuracy'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aggregate_entropy_accuracy.png', bbox_inches='tight', dpi=300)
    plt.close()

def aggregate_existing_results(base_dir, dataset_name, num_epochs, num_layers, model_name, seeds):
    """Aggregate results from existing seed directories."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    aggregate_dir = Path(f'{base_dir}/aggregate_results_{timestamp}')
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        'train': {'accuracy': [], 'loss': []},
        'val': {'accuracy': [], 'loss': []},
        'test': {'accuracy': [], 'loss': []}
    }
    
    all_depth_distributions = []
    all_entropy_accuracies = {'train': [], 'test': []}

    for seed in seeds:
        seed_dir = Path(f'{base_dir}/{dataset_name}_{num_epochs}epochs_{num_layers}layers_{model_name}_seed{seed}')
        
        try:
            # Load training metrics
            metrics_files = list(seed_dir.glob('baselinetraining_metrics_split_0_*.csv'))
            if metrics_files:
                metrics_df = pd.read_csv(metrics_files[0])
                for metric_type in ['train', 'val', 'test']:
                    all_metrics[metric_type]['accuracy'].append(metrics_df[f'{metric_type}_accuracy'].values)
                    all_metrics[metric_type]['loss'].append(metrics_df[f'{metric_type}_loss'].values)

            # Load depth analysis results
            for set_type in ['train', 'test']:
                depth_files = list(seed_dir.glob(f'full_depth_analysis_{set_type}_split_0_*.csv'))
                if depth_files:
                    depth_df = pd.read_csv(depth_files[0])
                    all_depth_distributions.append((seed, set_type, depth_df))
                
                entropy_files = list(seed_dir.glob(f'gnn_accuracy_vs_entropy_{set_type}_split_0_*.csv'))
                if entropy_files:
                    entropy_df = pd.read_csv(entropy_files[0])
                    all_entropy_accuracies[set_type].append(entropy_df)
        
        except Exception as e:
            print(f"Error processing seed {seed}: {str(e)}")
            continue

    if not all_metrics['train']['accuracy']:
        print("No valid metrics data found!")
        return

    # Create visualizations
    create_aggregate_training_plot(all_metrics, aggregate_dir)
    if all_depth_distributions:
        create_aggregate_depth_distribution(all_depth_distributions, aggregate_dir)
        create_aggregate_accuracy_depth_plots(all_depth_distributions, aggregate_dir)
    if any(all_entropy_accuracies.values()):
        create_aggregate_entropy_accuracy_plot(all_entropy_accuracies, aggregate_dir)

    print(f"Aggregate results saved in: {aggregate_dir}")

if __name__ == "__main__":
    # Example usage
    seeds = [3164711608, 894959334, 2487307261, 3349051410, 493067366]
    
    # Update these parameters according to your setup
    base_dir = "EntropyTracking"
    dataset_name = "Cora"  # or your dataset name
    num_epochs = 1500  # or your number of epochs
    num_layers = 5   # or your number of layers
    model_name = "SimpleGCN"  # or your model name
    
    aggregate_existing_results(
        base_dir=base_dir,
        dataset_name=dataset_name,
        num_epochs=num_epochs,
        num_layers=num_layers,
        model_name=model_name,
        seeds=seeds
    )