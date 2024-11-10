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

set_seed(3164711608)
def visualize_prediction_confidence_and_entropy(model, data, predictions, confidences, train_mask, test_mask, split_idx, true_labels, noise_level):
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

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 10))

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

        # Count nodes in each category
        category_counts = {
            'green': sum(1 for c in colors if c == 'green'),
            'red': sum(1 for c in colors if c == 'red')
        }

        # Count high entropy nodes in each category
        high_entropy_counts = {
            'green': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'green' and he),
            'red': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'red' and he)
        }

        # Create a custom legend with counts
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=f'Correct Predictions: {category_counts["green"]} (High Δ Entropy: {high_entropy_counts["green"]})', markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong Predictions: {category_counts["red"]} (High Δ Entropy: {high_entropy_counts["red"]})', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='*', color='w', label=f'High Δ Entropy: {sum(high_entropy_mask)}', markerfacecolor='gray', markersize=15),
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        filename_base = f'confidence_vs_entropy_{set_name.lower()}_split_{split_idx}_{timestamp}'
        plt.savefig(f'{filename_base}.png', bbox_inches='tight')
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
            df.to_csv(f'training_entropy_confidence_split_{split_idx}_{timestamp}.csv', index=False)

        return category_counts, high_entropy_counts

    # Plot for test set
    test_counts, test_high_entropy_counts = plot_set(test_mask.cpu(), "Test")

    # Plot for training set
    train_counts, train_high_entropy_counts = plot_set(train_mask.cpu(), "Training")

    return {
        'test': {'counts': test_counts, 'high_entropy_counts': test_high_entropy_counts},
        'train': {'counts': train_counts, 'high_entropy_counts': train_high_entropy_counts}
    }

# Assuming the kd_retention function is defined as provided
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
    



def determine_k_nodes(delta_entropy, train_mask, percentile, min_nodes=0, max_nodes=None):
    """
    Determine number of nodes to zero out based on entropy distribution.
    """
    # Special case: if percentile is None, return 0 nodes (baseline case)
    if percentile is None:
        print("\nBaseline mode: No nodes will be zeroed out")
        return 0, float('inf')
    
    # Get entropy values for training nodes only
    train_indices = train_mask.nonzero().squeeze().cpu().numpy()
    train_entropies = delta_entropy[train_indices]
    
    # Calculate threshold based on percentile
    threshold = np.percentile(train_entropies, percentile)
    
    # Count nodes above threshold
    nodes_above_threshold = np.sum(train_entropies >= threshold)
    
    # Use all nodes above threshold (with minimum safeguard)
    k_nodes = max(min_nodes, nodes_above_threshold)
    
    # Apply maximum limit if specified
    if max_nodes is not None:
        k_nodes = min(k_nodes, max_nodes)
    
    if k_nodes > 0:  # Only print if we're actually zeroing nodes
        print(f"\nEntropy Distribution Analysis:")
        print(f"Training nodes: {len(train_indices)}")
        print(f"Entropy threshold ({percentile}th percentile): {threshold:.4f}")
        print(f"Nodes above threshold: {nodes_above_threshold}")
        print(f"Selected k_nodes: {k_nodes} ({(k_nodes/len(train_indices)*100):.1f}% of training nodes)")
    
    return k_nodes, threshold

def train_and_get_results(data, model, optimizer,num_epochs, percentile,k_nodes=None, dynamic_zeroing=False):
    """
    Train the model with option to zero out high entropy nodes.
    
    Args:
        data: Input graph data
        model: The GNN model
        optimizer: The optimizer
        k_nodes: If None, automatically determine based on entropy distribution
        dynamic_zeroing: If True, zero out k nodes every epoch. If False, zero out k nodes once before training (default: False)
        percentile: Percentile threshold for determining high entropy nodes
    """
    # Add storage for entropy tracking

    num_nodes = data.x.size(0)
    entropy_history = np.zeros((num_nodes, num_epochs))
    
    avg_testacc_before = []
    avg_trainacc_before = []  # Added for training accuracy
    avg_acc_testallsplits_before = []
    avg_acc_trainallsplits_before = []  # Added for training accuracy
    avg_valacc_before = []
    avg_acc_valallsplits_before = []
    criterion = torch.nn.CrossEntropyLoss()

    def verify_zeroed_nodes(data_tensor, zeroed_indices, epoch=None):
        """Verify that specified nodes have been zeroed out correctly."""
        zeroed_features = data_tensor[zeroed_indices]
        is_zeroed = torch.all(zeroed_features == 0, dim=1)
        all_zeroed = torch.all(is_zeroed)
        
        if not all_zeroed:
            non_zeroed = zeroed_indices[~is_zeroed]
            print(f"WARNING: Not all specified nodes were zeroed out{'.' if epoch is None else f' at epoch {epoch}.'}")
            #print(f"Non-zeroed nodes: {non_zeroed.tolist()}")
            #print(f"Their values: {data_tensor[non_zeroed]}")
        else:
            print(f"✓ Successfully verified {len(zeroed_indices)} nodes were zeroed out" + 
                  (f" at epoch {epoch}" if epoch is not None else ""))
            
        return all_zeroed

    def train(model,optimizer):
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
            pred = out.argmax(dim=1)  # Use the class with highest probability. 
            val_correct = pred[val_mask] == data.y[val_mask]  # Check against ground-truth labels.
            val_acc = int(val_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
        return val_acc

    def test(model):
        model.eval()
        with torch.no_grad():
                    out= model(data.x, data.edge_index)
                    pred = out.argmax(dim=1)  # Use the class with highest probability. 
                    test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
                    test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc,pred,out
 
    # Add storage for all metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []

    for split_idx in range(0,1):
        model.reset_parameters()
        optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
        train_mask = data.train_mask[:,split_idx]
        test_mask = data.test_mask[:,split_idx]
        val_mask = data.val_mask[:,split_idx]
        
        #================ Check for data leakage =================#
        train_nodes = data.train_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        test_nodes = data.test_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        val_nodes = data.val_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        leakage_nodes = np.intersect1d(train_nodes, test_nodes)
        if len(leakage_nodes) > 0:
            print(f"Warning: Found {len(leakage_nodes)} nodes in both training and test sets. Stopping execution.")
            sys.exit(1)  # Exit the script due to data leakage
        if len(np.intersect1d(train_nodes, val_nodes)) > 0:
            print(f"Warning: Found {len(leakage_nodes)} nodes in both training and validation sets. Stopping execution.")   
            sys.exit(1)  # Exit the script due to data leakage
        
        # Reset training metrics for this split
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        test_losses = []
        test_accs = []
        
        print(f"Training for index = {split_idx}")

        # Initialize modified data
        modified_data = copy.deepcopy(data)
        high_entropy_nodes = []

        # For static zeroing, identify high entropy nodes before training
        if k_nodes is not None or not dynamic_zeroing:
            # Train for a few epochs to get initial entropy estimates
            warmup_epochs = 100
            print("Running warmup epochs to identify high entropy nodes...")
            for _ in range(warmup_epochs):
                model.train()
                optimizer.zero_grad()
                out = model(modified_data.x, modified_data.edge_index)
                loss = criterion(out[train_mask], modified_data.y[train_mask])
                loss.backward()
                optimizer.step()

            # Calculate initial delta entropy
            delta_entropy = kd_retention(model, modified_data, noise_level=1.0)
            
            # Determine k_nodes if not specified
            if k_nodes is None:
                k_nodes, threshold = determine_k_nodes(
                    delta_entropy, 
                    train_mask, 
                    percentile=percentile
                )
            
            # Get indices of k nodes with highest entropy among training nodes
            train_indices = train_mask.nonzero().squeeze().cpu().numpy()
            train_entropies = delta_entropy[train_indices]
            high_entropy_indices = train_indices[np.argsort(train_entropies)[-k_nodes:]]
            
            # Print entropy distribution visualization
            plt.figure(figsize=(10, 6))
            plt.hist(train_entropies, bins=50, alpha=0.7)
            plt.axvline(x=delta_entropy[high_entropy_indices].min(), 
                       color='r', linestyle='--', 
                       label=f'Zeroing threshold ({k_nodes} nodes)')
            plt.xlabel('Delta Entropy')
            plt.ylabel('Count')
            plt.title(f'Training Nodes Entropy Distribution (Split {split_idx})')
            plt.legend()
            plt.savefig(f'entropy_distribution_split_{split_idx}.png')
            plt.close()

            # Store high entropy nodes
            high_entropy_nodes = high_entropy_indices
            
            # Zero out the features for high entropy nodes
            modified_data.x[high_entropy_indices] = 0
            
            # Verify zeroing
            verify_zeroed_nodes(modified_data.x, high_entropy_indices)
            
            print(f"Zeroed out {k_nodes} nodes with highest initial entropy (static)")
            #print(f"Zeroed nodes: {high_entropy_indices.tolist()}")
            #print(f"Their initial entropy values: {delta_entropy[high_entropy_indices].tolist()}")
            
            # Reset model and optimizer after warmup
            model.reset_parameters()
            optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)

        # Add counters for dynamic zeroing verification
        if dynamic_zeroing:
            zeroing_changes = []  # Track which nodes are zeroed each epoch
            consecutive_same = 0  # Count epochs with same nodes being zeroed

        for epoch in tqdm(range(1, num_epochs+1)):
            # For dynamic zeroing, update zeroed nodes each epoch
            if k_nodes > 0 and dynamic_zeroing:
                # Store previous high entropy nodes for comparison
                prev_high_entropy_nodes = set(high_entropy_nodes) if len(high_entropy_nodes) > 0 else set()
                
                # Calculate delta entropy for this epoch
                delta_entropy = kd_retention(model, modified_data, noise_level=1.0)
                entropy_history[:, epoch-1] = delta_entropy
                
                # Reset modified data to original
                modified_data.x = data.x.clone()
                
                # Get indices of k nodes with highest entropy among training nodes
                train_indices = train_mask.nonzero().squeeze().cpu().numpy()
                train_entropies = delta_entropy[train_indices]
                high_entropy_indices = train_indices[np.argsort(train_entropies)[-k_nodes:]]
                
                # Store high entropy nodes for final epoch
                high_entropy_nodes = high_entropy_indices
                
                # Zero out the features for high entropy nodes
                modified_data.x[high_entropy_indices] = 0
                
                # Verify zeroing
                verify_zeroed_nodes(modified_data.x, high_entropy_indices, epoch)
                
                # Track changes in zeroed nodes
                current_nodes = set(high_entropy_indices.tolist())
                if prev_high_entropy_nodes == current_nodes:
                    consecutive_same += 1
                else:
                    consecutive_same = 0
                    zeroing_changes.append((epoch, prev_high_entropy_nodes, current_nodes))
                
                # Alert if nodes haven't changed for many epochs
                if consecutive_same > 50:  # Adjust threshold as needed
                    print(f"\nWarning: Same nodes have been zeroed for {consecutive_same} consecutive epochs")
                    print(f"Current zeroed nodes: {list(current_nodes)}")

            # Train with modified data
            model.train()
            optimizer.zero_grad()
            out = model(modified_data.x, modified_data.edge_index)
            loss = criterion(out[train_mask], modified_data.y[train_mask])
            loss.backward()
            optimizer.step()

            # Calculate and store training metrics
            with torch.no_grad():
                pred = out.argmax(dim=1)
                train_correct = pred[train_mask] == modified_data.y[train_mask]
                train_acc = float(train_correct.sum()) / int(train_mask.sum())
                train_accs.append(train_acc * 100)
                train_losses.append(float(loss))

                # Calculate validation metrics
                val_out = model(modified_data.x, modified_data.edge_index)
                val_loss = criterion(val_out[val_mask], modified_data.y[val_mask])
                val_pred = val_out.argmax(dim=1)
                val_correct = val_pred[val_mask] == modified_data.y[val_mask]
                val_acc = float(val_correct.sum()) / int(val_mask.sum())
                val_accs.append(val_acc * 100)
                val_losses.append(float(val_loss))

                # Calculate test metrics
                test_out = model(modified_data.x, modified_data.edge_index)
                test_loss = criterion(test_out[test_mask], modified_data.y[test_mask])
                test_pred = test_out.argmax(dim=1)
                test_correct = test_pred[test_mask] == modified_data.y[test_mask]
                test_acc = float(test_correct.sum()) / int(test_mask.sum())
                test_accs.append(test_acc * 100)
                test_losses.append(float(test_loss))

        # Plot training metrics with all data
        plot_training_metrics(
            train_losses, 
            train_accs,
            val_losses,
            val_accs,
            test_losses,
            test_accs,
            split_idx,
            k_nodes,
            dynamic_zeroing
        )

        # After training, print summary of dynamic zeroing
        if k_nodes > 0 and dynamic_zeroing:
            print("\nDynamic Zeroing Summary:")
            print(f"Total number of changes in zeroed nodes: {len(zeroing_changes)}")
            if len(zeroing_changes) > 0:
                print("\nSample of changes (first 5 and last 5 changes):")
                for i in range(min(5, len(zeroing_changes))):
                    epoch, old, new = zeroing_changes[i]
                    print(f"Epoch {epoch}: Changed from {list(old)} to {list(new)}")
                if len(zeroing_changes) > 10:
                    print("...")
                for i in range(max(5, len(zeroing_changes)-5), len(zeroing_changes)):
                    epoch, old, new = zeroing_changes[i]
                    print(f"Epoch {epoch}: Changed from {list(old)} to {list(new)}")

        # For final evaluation, use the original data
        model.eval()
        with torch.no_grad():
            # Use modified_data for training accuracy (to include zeroed nodes)
            out = model(modified_data.x, modified_data.edge_index)
            pred = out.argmax(dim=1)
            train_correct = pred[train_mask] == modified_data.y[train_mask]
            train_acc = int(train_correct.sum()) / int(train_mask.sum())
            avg_trainacc_before.append(train_acc*100)

            # Use original data for val/test (or modified_data - should be the same)
            val_acc = val(model)  
            avg_valacc_before.append(val_acc*100)  
            test_acc, pred, out = test(model)    
            avg_testacc_before.append(test_acc*100)   
            
            print(f'Training Accuracy (with zeroed nodes): {train_acc*100:.4f}')
            print(f'Val Accuracy: {val_acc*100:.4f}') 
            print(f'Test Accuracy: {test_acc*100:.4f}')
            print()        
        
        avg_acc_trainallsplits_before.append(np.mean(avg_trainacc_before))
        avg_acc_valallsplits_before.append(np.mean(avg_valacc_before))  
        avg_acc_testallsplits_before.append(np.mean(avg_testacc_before))            
    
        # Save zeroing information
        if k_nodes > 0:
            # Convert numpy types to native Python types
            zeroing_info = {
                'split_idx': int(split_idx),
                'k_nodes': int(k_nodes),
                'zeroing_type': 'dynamic' if dynamic_zeroing else 'static',
                'final_zeroed_nodes': [int(x) for x in high_entropy_nodes.tolist()],
                'final_entropy_values': [float(x) for x in delta_entropy[high_entropy_nodes].tolist()]
            }
            
            if dynamic_zeroing:
                zeroing_info.update({
                    'total_changes': int(len(zeroing_changes)),
                    'epochs_with_changes': [int(change[0]) for change in zeroing_changes],
                    'longest_unchanged_period': int(consecutive_same)
                })
            
            # Save to JSON
            import json
            with open(f'zeroing_info_split_{split_idx}_{("dynamic" if dynamic_zeroing else "static")}_{k_nodes}.json', 'w') as f:
                json.dump(zeroing_info, f, indent=4)
    
        visualize_prediction_confidence_and_entropy(
            model, data, pred, out.softmax(dim=1), train_mask, test_mask, split_idx, data.y, noise_level=1.0)
    
    return avg_acc_testallsplits_before, avg_acc_valallsplits_before, avg_acc_trainallsplits_before

def plot_training_metrics(losses, accuracies, val_losses, val_accuracies, test_losses, test_accuracies, 
                         split_idx, k_nodes, dynamic_zeroing):
    """
    Plot training, validation, and test metrics side by side.
    
    Args:
        losses: List of training losses
        accuracies: List of training accuracies
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
        test_losses: List of test losses
        test_accuracies: List of test accuracies
        split_idx: Current split index
        k_nodes: Number of nodes being zeroed
        dynamic_zeroing: Whether using dynamic or static zeroing
    """
    epochs = range(1, len(losses) + 1)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot losses
    ax1.plot(epochs, losses, 'b-', label='Training')
    ax1.plot(epochs, val_losses, 'g-', label='Validation')
    ax1.plot(epochs, test_losses, 'r-', label='Test')
    ax1.set_title(f'Loss vs Epochs (Split {split_idx})\n'
                 f'{"Dynamic" if dynamic_zeroing else "Static"} Zeroing, k={k_nodes}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(epochs, accuracies, 'b-', label='Training')
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation')
    ax2.plot(epochs, test_accuracies, 'r-', label='Test')
    ax2.set_title(f'Accuracy vs Epochs (Split {split_idx})\n'
                 f'{"Dynamic" if dynamic_zeroing else "Static"} Zeroing, k={k_nodes}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    zeroing_type = "dynamic" if dynamic_zeroing else "static"
    plt.savefig(f'training_metrics_split_{split_idx}_k{k_nodes}_{zeroing_type}.png', 
                bbox_inches='tight', 
                dpi=300)
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
    metrics_df.to_csv(f'training_metrics_split_{split_idx}_k{k_nodes}_{zeroing_type}.csv', 
                      index=False)
    
    # Print final metrics
    print(f"\nFinal metrics (Split {split_idx}):")
    print(f"Training - Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.2f}%")
    print(f"Validation - Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Test - Loss: {test_losses[-1]:.4f}, Accuracy: {test_accuracies[-1]:.2f}%")

# def plot_entropy_trajectories(entropy_history, train_mask, split_idx, predictions, true_labels, num_epochs):
#     """Plot delta entropy trajectories for training nodes."""
#     train_indices = train_mask.nonzero().squeeze().cpu().numpy()
    
#     # Create figure
#     plt.figure(figsize=(15, 10))
    
#     # Get correctness of predictions for training nodes
#     correct_predictions = predictions[train_mask] == true_labels[train_mask]
    
#     # Plot trajectories
#     epochs = np.arange(1, num_epochs + 1)
    
#     # Plot correctly classified nodes
#     correct_nodes = train_indices[correct_predictions.cpu().numpy()]
#     for node_idx in correct_nodes:
#         plt.plot(epochs, entropy_history[node_idx], 'g-', alpha=0.1)
    
#     # Plot incorrectly classified nodes
#     incorrect_nodes = train_indices[~correct_predictions.cpu().numpy()]
#     for node_idx in incorrect_nodes:
#         plt.plot(epochs, entropy_history[node_idx], 'r-', alpha=0.1)
    
#     # Calculate and plot mean trajectories
#     mean_correct = entropy_history[correct_nodes].mean(axis=0)
#     mean_incorrect = entropy_history[incorrect_nodes].mean(axis=0)
    
#     plt.plot(epochs, mean_correct, 'g-', linewidth=2, label='Mean (Correct)')
#     plt.plot(epochs, mean_incorrect, 'r-', linewidth=2, label='Mean (Incorrect)')
    
#     plt.xlabel('Epoch')
#     plt.ylabel('Delta Entropy')
#     plt.title(f'Delta Entropy Trajectories for Training Nodes (Split {split_idx})')
#     plt.legend([
#         f'Correct Predictions (n={len(correct_nodes)})',
#         f'Incorrect Predictions (n={len(incorrect_nodes)})'
#     ])
    
#     # Save plot
#     plt.savefig(f'entropy_trajectories_split_{split_idx}.png', bbox_inches='tight', dpi=300)
#     plt.close()
    
#     # Save trajectory data
#     trajectory_data = pd.DataFrame({
#         'epoch': np.tile(epochs, len(train_indices)),
#         'node_idx': np.repeat(train_indices, num_epochs),
#         'delta_entropy': entropy_history[train_indices].flatten(),
#         'correct_prediction': np.repeat(correct_predictions.cpu().numpy(), num_epochs)
#     })
#     trajectory_data.to_csv(f'entropy_trajectories_split_{split_idx}.csv', index=False)

# def plot_high_entropy_node_trajectories(entropy_history, train_mask, split_idx, predictions, true_labels, num_epochs):
#     """Plot individual trajectories for high entropy nodes."""
#     # Move tensors to CPU first
#     train_indices = train_mask.nonzero().squeeze().cpu().numpy()
#     predictions = predictions.cpu()
#     true_labels = true_labels.cpu()
    
#     # Get final epoch entropy values
#     final_entropy = entropy_history[:, -1]
    
#     # Identify high entropy nodes (top 10%) among training nodes
#     train_final_entropy = final_entropy[train_indices]
#     high_entropy_threshold = np.percentile(train_final_entropy, 90)
#     high_entropy_mask = train_final_entropy >= high_entropy_threshold
#     high_entropy_indices = train_indices[high_entropy_mask]
    
#     # Create individual plots for each high entropy node
#     for node_idx in high_entropy_indices:
#         plt.figure(figsize=(10, 6))
        
#         # Plot the trajectory
#         epochs = np.arange(1, num_epochs + 1)
#         node_trajectory = entropy_history[node_idx]
        
#         # Determine if the node was correctly classified
#         is_correct = predictions[node_idx] == true_labels[node_idx]
#         color = 'g' if is_correct else 'r'
#         label = 'Correctly Classified' if is_correct else 'Incorrectly Classified'
        
#         plt.plot(epochs, node_trajectory, color=color, linewidth=2, label=label)
        
#         # Add horizontal line for high entropy threshold
#         plt.axhline(y=high_entropy_threshold, color='gray', linestyle='--', 
#                    label='High Entropy Threshold')
        
#         plt.xlabel('Epoch')
#         plt.ylabel('Delta Entropy')
#         plt.title(f'Delta Entropy Trajectory for High Entropy Node {node_idx} (Split {split_idx})')
#         plt.legend()
        
#         # Save plot
#         plt.savefig(f'high_entropy_node_{node_idx}_split_{split_idx}.png', 
#                    bbox_inches='tight', dpi=300)
#         plt.close()
    
#     # Save data for high entropy nodes
#     high_entropy_data = pd.DataFrame({
#         'node_idx': high_entropy_indices,
#         'final_entropy': final_entropy[high_entropy_indices],
#         'correct_prediction': (predictions[high_entropy_indices] == true_labels[high_entropy_indices]).numpy(),
#         'predicted_label': predictions[high_entropy_indices].numpy(),
#         'true_label': true_labels[high_entropy_indices].numpy()
#     })
#     high_entropy_data.to_csv(f'high_entropy_nodes_summary_split_{split_idx}.csv', index=False)
    
#     # Create summary trajectory data for high entropy nodes
#     high_entropy_trajectories = pd.DataFrame({
#         'epoch': np.tile(epochs, len(high_entropy_indices)),
#         'node_idx': np.repeat(high_entropy_indices, num_epochs),
#         'delta_entropy': entropy_history[high_entropy_indices].flatten(),
#         'correct_prediction': np.repeat(
#             (predictions[high_entropy_indices] == true_labels[high_entropy_indices]).numpy(),
#             num_epochs
#         )
#     })
#     high_entropy_trajectories.to_csv(f'high_entropy_trajectories_split_{split_idx}.csv', index=False)















