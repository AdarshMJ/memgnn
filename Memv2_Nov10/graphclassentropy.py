import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from pathlib import Path

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_save_dir(dataset_name, num_epochs, num_layers, model_name, seed):
    base_dir = Path(f'results/{dataset_name}/{model_name}_{num_layers}layers_{num_epochs}epochs')
    save_dir = base_dir / f'seed_{seed}'
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

def get_predictions(model, loader):
    """Get model predictions for a dataset"""
    model.eval()
    predictions = []
    confidences = []
    true_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            conf = F.softmax(out, dim=1)
            
            predictions.extend(pred.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
            true_labels.extend(batch.y.cpu().numpy())
    
    return np.array(predictions), np.array(confidences), np.array(true_labels)

def create_knn_probe_graphs(features, labels, n_neighbors=3):
    """Create and fit a KNN probe for graph features"""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(features, labels)
    return knn

def get_prediction_depth(prediction_history, final_prediction):
    """Calculate prediction depth for a graph"""
    for depth, pred in enumerate(prediction_history):
        if pred == final_prediction:
            return depth
    return len(prediction_history) - 1

def analyze_prediction_depth_graphs(model, loader, predictions, confidences, true_labels, save_dir, set_name):
    """Analyze prediction depth for graph classification."""
    device = next(model.parameters()).device
    graph_info = {}
    
    # Collect all representations and labels first
    all_layer_representations = [[] for _ in range(len(model.layers))]
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Get representations at each layer
            _, layer_representations = model(batch.x, batch.edge_index, batch.batch, return_layer_rep=True)
            
            # Store representations and labels
            for layer_idx, layer_repr in enumerate(layer_representations):
                all_layer_representations[layer_idx].append(layer_repr.detach().cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # Concatenate all representations
    all_layer_representations = [np.concatenate(layer_repr, axis=0) for layer_repr in all_layer_representations]
    all_labels = np.array(all_labels)
    
    # Now process each graph
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        _, layer_representations = model(batch.x, batch.edge_index, batch.batch, return_layer_rep=True)
        
        for graph_idx in range(batch.num_graphs):
            prediction_history = []
            knn_correct_history = []
            
            # Get predictions at each layer using KNN probe
            for layer_idx, layer_repr in enumerate(layer_representations):
                current_layer_repr = layer_repr[graph_idx].detach().cpu().numpy().reshape(1, -1)
                knn = create_knn_probe_graphs(all_layer_representations[layer_idx], all_labels)
                layer_pred = knn.predict(current_layer_repr)[0]
                prediction_history.append(layer_pred)
                knn_correct_history.append(int(layer_pred == batch.y[graph_idx].item()))  # Convert to int
            
            # Store information for this graph
            global_idx = batch_idx * loader.batch_size + graph_idx
            graph_info[global_idx] = {
                'prediction_depth': get_prediction_depth(prediction_history, predictions[global_idx]),
                'final_prediction': predictions[global_idx],
                'true_label': true_labels[global_idx],
                'knn_correct_history': knn_correct_history,  # Now storing as list directly
                'gnn_correct': predictions[global_idx] == true_labels[global_idx]
            }
    
    # Create DataFrame and visualize results
    results_df = pd.DataFrame.from_dict(graph_info, orient='index')
    visualize_prediction_depth_graphs(results_df, save_dir, set_name, model)
    
    return results_df

def visualize_prediction_depth_graphs(df, save_dir, set_name, model):
    """Create visualizations for prediction depth analysis."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Calculate accuracies by depth
    depth_groups = df.groupby('prediction_depth')
    gnn_acc = depth_groups['gnn_correct'].mean()
    
    # Fix KNN accuracy calculation - now handling lists directly
    knn_acc = depth_groups.apply(lambda x: np.mean([np.mean(hist) for hist in x['knn_correct_history']]))
    
    counts = depth_groups.size()
    
    # Get layer names
    num_layers = len(model.layers)
    layer_names = [f"Layer {i+1}" for i in range(num_layers)]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # GNN Accuracy plot
    ax1.bar(range(len(counts)), counts, alpha=0.3, color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(gnn_acc)), gnn_acc * 100, 'r-o')
    
    # Set x-axis ticks and labels
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    
    ax1.set_title(f'GNN Accuracy vs Depth ({set_name} Set)')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Graphs', color='blue')
    ax1_twin.set_ylabel('Accuracy (%)', color='red')
    ax1_twin.set_ylim(-5, 105)
    ax1_twin.grid(True, alpha=0.3)
    
    # Add legend
    ax1.legend(['Number of Graphs'], loc='upper left')
    ax1_twin.legend(['GNN Accuracy'], loc='upper right')
    
    # KNN Accuracy plot
    ax2.bar(range(len(counts)), counts, alpha=0.3, color='blue')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(knn_acc)), knn_acc * 100, 'r-o')
    
    # Set x-axis ticks and labels
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45, ha='right')
    
    ax2.set_title(f'KNN Probe Accuracy vs Depth ({set_name} Set)')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Number of Graphs', color='blue')
    ax2_twin.set_ylabel('Accuracy (%)', color='red')
    ax2_twin.set_ylim(-5, 105)
    ax2_twin.grid(True, alpha=0.3)
    
    # Add legend
    ax2.legend(['Number of Graphs'], loc='upper left')
    ax2_twin.legend(['KNN Accuracy'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'accuracy_vs_depth_{set_name}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics for {set_name} Set:")
    print("Number of graphs at each depth:")
    for depth, count in counts.items():
        print(f"Layer {depth+1}: {count} graphs")
    print("\nAccuracies at each depth:")
    for depth in range(len(layer_names)):
        if depth in gnn_acc.index:
            print(f"Layer {depth+1}:")
            print(f"  GNN Accuracy: {float(gnn_acc[depth])*100:.2f}%")
            print(f"  KNN Accuracy: {float(knn_acc[depth])*100:.2f}%")

def train_and_get_results(train_loader, val_loader, test_loader, model, optimizer, 
                         num_epochs, dataset_name, num_layers, seed):
    """Train model and analyze final prediction depth."""
    model_name = model.__class__.__name__
    save_dir = get_save_dir(dataset_name, num_epochs, num_layers, model_name, seed)
    criterion = torch.nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    
    # Initialize metrics dictionary
    metrics = {
        'train': {'accuracy': [], 'loss': []},
        'val': {'accuracy': [], 'loss': []},
        'test': {'accuracy': [], 'loss': []}
    }
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(dim=1)
            train_correct += int((pred == batch.y).sum())
            train_total += batch.num_graphs
            train_loss += float(loss) * batch.num_graphs
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        metrics['train']['loss'].append(epoch_train_loss)
        metrics['train']['accuracy'].append(epoch_train_acc)
        
        # Validation metrics
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                pred = out.argmax(dim=1)
                
                val_correct += int((pred == batch.y).sum())
                val_total += batch.num_graphs
                val_loss += float(loss) * batch.num_graphs
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        metrics['val']['loss'].append(epoch_val_loss)
        metrics['val']['accuracy'].append(epoch_val_acc)
        
        # Test metrics
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                pred = out.argmax(dim=1)
                
                test_correct += int((pred == batch.y).sum())
                test_total += batch.num_graphs
                test_loss += float(loss) * batch.num_graphs
        
        epoch_test_loss = test_loss / test_total
        epoch_test_acc = test_correct / test_total
        metrics['test']['loss'].append(epoch_test_loss)
        metrics['test']['accuracy'].append(epoch_test_acc)
    
    # After training, analyze prediction depth
    model.eval()
    depth_results = {}
    
    for name, loader in [('train', train_loader), ('test', test_loader)]:
        predictions, confidences, true_labels = get_predictions(model, loader)
        depth_df = analyze_prediction_depth_graphs(
            model=model,
            loader=loader,
            predictions=predictions,
            confidences=confidences,
            true_labels=true_labels,
            save_dir=save_dir,
            set_name=name
        )
        depth_results[name] = depth_df
    
    return metrics, depth_results

def plot_training_metrics(metrics, save_dir):
    """Plot training, validation, and test metrics with vertical error bars."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Calculate mean and std across seeds
    mean_metrics = {
        split: {
            metric: np.mean(values, axis=0) 
            for metric, values in split_metrics.items()
        }
        for split, split_metrics in metrics.items()
    }
    std_metrics = {
        split: {
            metric: np.std(values, axis=0) 
            for metric, values in split_metrics.items()
        }
        for split, split_metrics in metrics.items()
    }
    
    # Plot accuracy with vertical error bars
    plt.figure(figsize=(10, 6))
    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    
    for split in ['train', 'val', 'test']:
        mean_acc = mean_metrics[split]['accuracy']
        std_acc = std_metrics[split]['accuracy']
        epochs = range(1, len(mean_acc) + 1)
        
        plt.errorbar(epochs, mean_acc, yerr=std_acc,  # Changed from 1.96 * std_acc to std_acc for raw standard deviation
                    label=f'{split.capitalize()} Accuracy',
                    color=colors[split], capsize=3, capthick=1,
                    elinewidth=1, markeredgewidth=1)
    
    plt.title('Model Accuracy Over Time (with std dev)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'accuracy_plot_{timestamp}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Similar changes for loss plot...

def create_aggregate_depth_analysis(all_depth_distributions, save_dir):
    """Create aggregate analysis of prediction depth across all seeds with error bars."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Separate train and test results
    train_results = []
    test_results = []
    
    for seed, split, df in all_depth_distributions:
        if split == 'train':
            train_results.append(df)
        else:
            test_results.append(df)
    
    def plot_aggregate_depth_metrics(results, set_name):
        depth_metrics = []
        
        for df in results:
            depth_groups = df.groupby('prediction_depth')
            gnn_acc = depth_groups['gnn_correct'].mean()
            knn_acc = depth_groups.apply(lambda x: np.mean([np.mean(hist) for hist in x['knn_correct_history']]))
            counts = depth_groups.size()
            depth_metrics.append({'gnn_acc': gnn_acc, 'knn_acc': knn_acc, 'counts': counts})
        
        max_depth = max(max(m['gnn_acc'].index) for m in depth_metrics)
        depths = range(max_depth + 1)
        
        # Calculate means
        gnn_means = np.array([np.mean([m['gnn_acc'].get(d, 0) for m in depth_metrics]) for d in depths])
        knn_means = np.array([np.mean([m['knn_acc'].get(d, 0) for m in depth_metrics]) for d in depths])
        count_means = np.array([np.mean([m['counts'].get(d, 0) for m in depth_metrics]) for d in depths])
        
        # Calculate standard errors
        n_seeds = len(results)
        gnn_stderrs = np.array([np.std([m['gnn_acc'].get(d, 0) for m in depth_metrics]) / np.sqrt(n_seeds) for d in depths])
        knn_stderrs = np.array([np.std([m['knn_acc'].get(d, 0) for m in depth_metrics]) / np.sqrt(n_seeds) for d in depths])
        
        # Calculate 95% CI
        gnn_ci = 1.96 * gnn_stderrs
        knn_ci = 1.96 * knn_stderrs
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # GNN Accuracy plot with error bars
        ax1.bar(depths, count_means, alpha=0.3, color='blue', label='Number of Graphs')
        ax1_twin = ax1.twinx()
        ax1_twin.errorbar(depths, gnn_means * 100, yerr=gnn_ci * 100,
                         fmt='r-o', label='GNN Accuracy',
                         capsize=3, capthick=1, elinewidth=1)
        
        ax1.set_title(f'GNN Accuracy vs Depth ({set_name} Set)')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Average Number of Graphs', color='blue')
        ax1_twin.set_ylabel('Accuracy (%)', color='red')
        ax1_twin.set_ylim(-5, 105)
        ax1_twin.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # KNN Accuracy plot with error bars
        ax2.bar(depths, count_means, alpha=0.3, color='blue', label='Number of Graphs')
        ax2_twin = ax2.twinx()
        ax2_twin.errorbar(depths, knn_means * 100, yerr=knn_ci * 100,
                         fmt='r-o', label='KNN Accuracy',
                         capsize=3, capthick=1, elinewidth=1)
        
        ax2.set_title(f'KNN Probe Accuracy vs Depth ({set_name} Set)')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Average Number of Graphs', color='blue')
        ax2_twin.set_ylabel('Accuracy (%)', color='red')
        ax2_twin.set_ylim(-5, 105)
        ax2_twin.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'aggregate_accuracy_vs_depth_{set_name}_{timestamp}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print statistics
        print(f"\n{set_name} Set Statistics:")
        for d in depths:
            print(f"\nLayer {d}:")
            print(f"Average number of graphs: {count_means[d]:.1f}")
            print(f"GNN Accuracy: {gnn_means[d]*100:.1f}% ± {gnn_ci[d]*100:.1f}%")
            print(f"KNN Accuracy: {knn_means[d]*100:.1f}% ± {knn_ci[d]*100:.1f}%")
    
    # Create aggregate plots for both train and test sets
    plot_aggregate_depth_metrics(train_results, 'Train')
    plot_aggregate_depth_metrics(test_results, 'Test')

def run_multiple_seeds(train_loader, val_loader, test_loader, model_class, optimizer_class, 
                      optimizer_params, num_epochs, dataset_name, num_layers, seeds):
    all_metrics = {
        'train': {'accuracy': [], 'loss': []},
        'val': {'accuracy': [], 'loss': []},
        'test': {'accuracy': [], 'loss': []}
    }
    all_depth_distributions = []
    
    # Store per-seed results
    seed_results = []
    
    for seed in seeds:
        print(f"\nTraining with seed {seed}")
        set_seed(seed)
        
        model = model_class()
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        
        metrics, depth_results = train_and_get_results(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            optimizer=optimizer,
            num_epochs=num_epochs,
            dataset_name=dataset_name,
            num_layers=num_layers,
            seed=seed
        )
        
        # Store final accuracies for this seed
        seed_result = {
            'seed': seed,
            'train_acc': metrics['train']['accuracy'][-1],
            'val_acc': metrics['val']['accuracy'][-1],
            'test_acc': metrics['test']['accuracy'][-1],
            'train_loss': metrics['train']['loss'][-1],
            'val_loss': metrics['val']['loss'][-1],
            'test_loss': metrics['test']['loss'][-1]
        }
        seed_results.append(seed_result)
        
        # Print final accuracies for this seed
        print(f"\nFinal accuracies for seed {seed}:")
        print(f"Train accuracy: {seed_result['train_acc']*100:.2f}%")
        print(f"Val accuracy:   {seed_result['val_acc']*100:.2f}%")
        print(f"Test accuracy:  {seed_result['test_acc']*100:.2f}%")
        print("-" * 50)
        
        # Store metrics for plotting
        for split in ['train', 'val', 'test']:
            all_metrics[split]['accuracy'].append(metrics[split]['accuracy'])
            all_metrics[split]['loss'].append(metrics[split]['loss'])
        
        for split in ['train', 'test']:
            all_depth_distributions.append((seed, split, depth_results[split]))
    
    # Calculate summary statistics
    train_accs = [result['train_acc'] for result in seed_results]
    val_accs = [result['val_acc'] for result in seed_results]
    test_accs = [result['test_acc'] for result in seed_results]
    
    mean_train = np.mean(train_accs)
    std_train = np.std(train_accs)
    mean_val = np.mean(val_accs)
    std_val = np.std(val_accs)
    mean_test = np.mean(test_accs)
    std_test = np.std(test_accs)
    
    # Print summary statistics
    print("\nSummary Statistics (averaged over all seeds):")
    print(f"Final Train Accuracy: {mean_train*100:.2f}% ± {std_train*100:.2f}%")
    print(f"Final Val Accuracy:   {mean_val*100:.2f}% ± {std_val*100:.2f}%")
    print(f"Final Test Accuracy:  {mean_test*100:.2f}% ± {std_test*100:.2f}%")
    
    # Save results to CSV
    base_save_dir = Path(f'results/{dataset_name}/aggregate_analysis')
    base_save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save per-seed results
    seed_df = pd.DataFrame(seed_results)
    seed_df.to_csv(base_save_dir / f'per_seed_results_{timestamp}.csv', index=False)
    
    # Save summary statistics
    summary_stats = {
        'metric': ['train_accuracy', 'val_accuracy', 'test_accuracy'],
        'mean': [mean_train, mean_val, mean_test],
        'std': [std_train, std_val, std_test]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(base_save_dir / f'summary_stats_{timestamp}.csv', index=False)
    
    # Plot aggregate metrics with vertical error bars
    plot_training_metrics(all_metrics, base_save_dir)
    create_aggregate_depth_analysis(all_depth_distributions, base_save_dir)
    
    return mean_train, std_train, mean_test, std_test














