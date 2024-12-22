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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
    knn_acc = depth_groups.apply(lambda x: np.mean([np.mean(hist) for hist in x['knn_correct_history']]))
    counts = depth_groups.size()
    
    # Get actual layer names from the model
    layer_names = [layer.__class__.__name__ for layer in model.layers]
    positions = np.arange(len(layer_names))
    
    # Ensure data arrays align with positions
    counts_array = np.zeros(len(positions))
    gnn_acc_array = np.zeros(len(positions))
    knn_acc_array = np.zeros(len(positions))
    
    for depth in range(len(positions)):
        if depth in counts.index:
            counts_array[depth] = counts[depth]
            gnn_acc_array[depth] = gnn_acc[depth]
            knn_acc_array[depth] = knn_acc[depth]
    
    # Increase default font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # GNN Accuracy plot
    ax1.bar(positions, counts_array, alpha=0.3, color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(positions, gnn_acc_array * 100, 'r-o')
    
    # Set x-axis ticks and labels for GNN plot
    ax1.set_xticks(positions)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=12)
    
    ax1.set_title(f'GNN Accuracy vs Depth ({set_name} Set)', fontsize=16)
    ax1.set_xlabel('Layer Type', fontsize=14)
    ax1.set_ylabel('Number of Graphs', color='blue', fontsize=14)
    ax1_twin.set_ylabel('Accuracy (%)', color='red', fontsize=14)
    
    # KNN Accuracy plot
    ax2.bar(positions, counts_array, alpha=0.3, color='blue')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(positions, knn_acc_array * 100, 'g-o')
    
    # Set x-axis ticks and labels for KNN plot
    ax2.set_xticks(positions)
    ax2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=12)
    
    ax2.set_title(f'KNN Probe Accuracy vs Depth ({set_name} Set)', fontsize=16)
    ax2.set_xlabel('Layer Type', fontsize=14)
    ax2.set_ylabel('Number of Graphs', color='blue', fontsize=14)
    ax2_twin.set_ylabel('Accuracy (%)', color='green', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'depth_analysis_{set_name}_{timestamp}.png', bbox_inches='tight', dpi=300)
    plt.close()

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

def create_aggregate_depth_analysis(all_depth_distributions, save_dir, model):
    """Create aggregate analysis of prediction depth across all seeds with error bars."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get actual layer names based on model structure
    layer_names = [f'GraphConv_{i}' for i in range(len(model.layers))]
    positions = np.arange(len(layer_names))
    
    # Separate train and test results
    train_results = []
    test_results = []
    
    for seed, split, df in all_depth_distributions:
        if split == 'train':
            train_results.append(df)
        else:
            test_results.append(df)
    
    # Calculate metrics for both train and test sets
    def get_metrics(results):
        depth_metrics = []
        for df in results:
            depth_groups = df.groupby('prediction_depth')
            gnn_acc = depth_groups['gnn_correct'].mean()
            knn_acc = depth_groups.apply(lambda x: np.mean([np.mean(hist) for hist in x['knn_correct_history']]))
            counts = depth_groups.size()
            
            # Ensure all positions are covered
            full_gnn_acc = np.zeros(len(positions))
            full_knn_acc = np.zeros(len(positions))
            full_counts = np.zeros(len(positions))
            
            for depth in range(len(positions)):
                if depth in gnn_acc.index:
                    full_gnn_acc[depth] = gnn_acc[depth]
                    full_knn_acc[depth] = knn_acc[depth]
                    full_counts[depth] = counts[depth]
            
            depth_metrics.append({
                'gnn_acc': full_gnn_acc,
                'knn_acc': full_knn_acc,
                'counts': full_counts
            })
        return depth_metrics
    
    train_metrics = get_metrics(train_results)
    test_metrics = get_metrics(test_results)
    
    # Calculate statistics
    def calculate_stats(metrics):
        n_seeds = len(metrics)
        gnn_means = np.mean([m['gnn_acc'] for m in metrics], axis=0)
        knn_means = np.mean([m['knn_acc'] for m in metrics], axis=0)
        count_means = np.mean([m['counts'] for m in metrics], axis=0)
        
        gnn_stderrs = np.std([m['gnn_acc'] for m in metrics], axis=0) / np.sqrt(n_seeds)
        knn_stderrs = np.std([m['knn_acc'] for m in metrics], axis=0) / np.sqrt(n_seeds)
        
        return {
            'gnn_means': gnn_means,
            'knn_means': knn_means,
            'count_means': count_means,
            'gnn_ci': 1.96 * gnn_stderrs,
            'knn_ci': 1.96 * knn_stderrs
        }
    
    train_stats = calculate_stats(train_metrics)
    test_stats = calculate_stats(test_metrics)
    
    # Create plots
    plt.rcParams.update({'font.size': 14})
    
    # Plot GNN accuracies (Train and Test)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # GNN Accuracies
    ax1.bar(positions, train_stats['count_means'], alpha=0.3, color='blue')
    ax1_twin = ax1.twinx()
    
    # Plot both train and test GNN accuracies
    ax1_twin.errorbar(positions, train_stats['gnn_means'] * 100, 
                     yerr=train_stats['gnn_ci'] * 100,
                     fmt='r-o', capsize=3, capthick=1, elinewidth=1, label='Train')
    ax1_twin.errorbar(positions, test_stats['gnn_means'] * 100, 
                     yerr=test_stats['gnn_ci'] * 100,
                     fmt='b-o', capsize=3, capthick=1, elinewidth=1, label='Test')
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=12)
    ax1.set_title('GNN Accuracy vs Depth', fontsize=16)
    ax1.set_xlabel('Layer', fontsize=14)
    ax1.set_ylabel('Number of Graphs', color='blue', fontsize=14)
    ax1_twin.set_ylabel('Accuracy (%)', color='black', fontsize=14)
    ax1_twin.legend()
    
    # KNN Accuracies
    ax2.bar(positions, train_stats['count_means'], alpha=0.3, color='blue')
    ax2_twin = ax2.twinx()
    
    # Plot both train and test KNN accuracies
    ax2_twin.errorbar(positions, train_stats['knn_means'] * 100,
                     yerr=train_stats['knn_ci'] * 100,
                     fmt='r-o', capsize=3, capthick=1, elinewidth=1, label='Train')
    ax2_twin.errorbar(positions, test_stats['knn_means'] * 100,
                     yerr=test_stats['knn_ci'] * 100,
                     fmt='b-o', capsize=3, capthick=1, elinewidth=1, label='Test')
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=12)
    ax2.set_title('KNN Probe Accuracy vs Depth', fontsize=16)
    ax2.set_xlabel('Layer', fontsize=14)
    ax2.set_ylabel('Number of Graphs', color='blue', fontsize=14)
    ax2_twin.set_ylabel('Accuracy (%)', color='black', fontsize=14)
    ax2_twin.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / f'aggregate_depth_analysis_{timestamp}.png', bbox_inches='tight', dpi=300)
    plt.close()

def run_multiple_seeds(train_loader, val_loader, test_loader, model_class, optimizer_class, 
                      optimizer_params, num_epochs, dataset_name, num_layers, seeds):
    all_metrics = {
        'train': {'accuracy': [], 'loss': []},
        'val': {'accuracy': [], 'loss': []},
        'test': {'accuracy': [], 'loss': []}
    }
    
    test_predictions_by_seed = {}
    all_depth_distributions = []
    seed_results = []
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Training with seed {seed}")
        print(f"{'='*50}")
        
        set_seed(seed)
        device = train_loader.dataset[0].x.device
        model = model_class().to(device)
        model.reset_parameters()
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
        
        # Store test predictions
        model.eval()
        test_preds = []
        test_true = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                test_preds.extend(pred.cpu().tolist())
                test_true.extend(batch.y.cpu().tolist())
        
        # test_predictions_by_seed[seed] = {
        #     'predictions': test_preds,
        #     'true_labels': test_true
        # }
        
        #print(f"\nTest Set Analysis for seed {seed}:")
       # print(f"First 10 predictions: {test_preds[:10]}")
        #print(f"First 10 true labels: {test_true[:10]}")
        
        # Compare with previous seeds
        # if len(test_predictions_by_seed) > 1:
        #     prev_seed = list(test_predictions_by_seed.keys())[-2]
        #     prev_preds = test_predictions_by_seed[prev_seed]['predictions']
        #     curr_preds = test_predictions_by_seed[seed]['predictions']
        #     num_different = sum(p1 != p2 for p1, p2 in zip(prev_preds, curr_preds))
        #     print(f"\nNumber of different predictions from previous seed: {num_different}")
        
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
        print(f"\nFinal accuracies for seed {seed}:")
        print(f"Train accuracy: {seed_result['train_acc']*100:.2f}%")
        print(f"Val accuracy:   {seed_result['val_acc']*100:.2f}%")
        print(f"Test accuracy:  {seed_result['test_acc']*100:.2f}%")
        print("-" * 50)
        
        # Store seed results
        seed_results.append(seed_result)
        
        # Store metrics for plotting
        for split in ['train', 'val', 'test']:
            all_metrics[split]['accuracy'].append(metrics[split]['accuracy'])
            all_metrics[split]['loss'].append(metrics[split]['loss'])
        
        # Store depth results
        for split in ['train', 'test']:
            if split in depth_results:
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
    base_save_dir = Path(f'results/{dataset_name}/_aggregate_analysis')
    base_save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save per-seed results
    seed_df = pd.DataFrame([{
        'seed': seed,
        'train_acc': acc,
        'val_acc': val_accs[i],
        'test_acc': test_accs[i]
    } for i, (seed, acc) in enumerate(zip(seeds, train_accs))])
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
    create_aggregate_depth_analysis(all_depth_distributions, base_save_dir, model)
    
    return mean_train, std_train, mean_test, std_test














