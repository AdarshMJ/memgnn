

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
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#planetoid_val_seeds =  [3164711608]
#planetoid_val_seeds = [3164711608,894959334,2487307261,3349051410,493067366]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def visualize_prediction_confidence_and_entropy(model, data, predictions, confidences, train_mask, test_mask, split_idx, true_labels, noise_level=0.1):
    # Calculate KD retention (assuming this function is defined elsewhere)
    delta_entropy = kd_retention(model, data, noise_level)

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
        plt.savefig(f'confidence_vs_entropy_{set_name.lower()}_split_{split_idx}2.png', bbox_inches='tight')
        plt.close()

        # Print the counts
        print(f"Split {split_idx} {set_name} Set Node Counts:")
        print(f"Correct Predictions: {category_counts['green']} (High Δ Entropy: {high_entropy_counts['green']})")
        print(f"Wrong Predictions: {category_counts['red']} (High Δ Entropy: {high_entropy_counts['red']})")
        print(f"Total High Δ Entropy: {sum(high_entropy_mask)}")
        print()

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
    



# def entropy_node_dropout(data: Data, model, dropout_rate: float, noise_level: float):
#     original_num_nodes = data.num_nodes
#     original_num_edges = data.edge_index.shape[1]
    
#     delta_entropy = kd_retention(model, data, noise_level)
#     num_nodes = data.num_nodes
#     num_drop = int(num_nodes * dropout_rate)
    
#     # Sort nodes by delta entropy and get indices of nodes to drop
#     drop_indices = np.argsort(delta_entropy)[-num_drop:]
    
#     # Create a mask for nodes to keep
#     keep_mask = torch.ones(num_nodes, dtype=torch.bool)
#     keep_mask[drop_indices] = False
    
#     # Apply the mask to node features and edge indices
#     keep_mask = keep_mask.cpu()
#     new_x = data.x[keep_mask]
#     new_edge_index = data.edge_index[:, (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)]
#     new_edge_index = new_edge_index.cpu()
#     new_edge_index = new_edge_index[:, keep_mask[new_edge_index[0]] & keep_mask[new_edge_index[1]]]
    
#     # Update node indices in edge_index
#     node_map = torch.cumsum(keep_mask, dim=0) - 1
#     new_edge_index = node_map[new_edge_index]
    
#     # Create a new Data object with dropped nodes
#     new_data = Data(x=new_x, edge_index=new_edge_index)
    
#     # Verification checks
#     nodes_dropped = original_num_nodes - new_data.num_nodes
#     edges_dropped = original_num_edges - new_data.edge_index.shape[1]
    
#     logging.info(f"Nodes dropped: {nodes_dropped} ({nodes_dropped/original_num_nodes:.2%})")
#     logging.info(f"Edges dropped: {edges_dropped} ({edges_dropped/original_num_edges:.2%})")
    
#     if abs(nodes_dropped - num_drop) > 1:  # Allow for small rounding differences
#         logging.warning(f"Unexpected number of nodes dropped. Expected: {num_drop}, Actual: {nodes_dropped}")
    
#     if new_data.num_nodes >= original_num_nodes:
#         logging.error("No nodes were dropped!")
#         raise ValueError("Node dropping failed: no nodes were removed")
#     print(new_data)
#     return new_data.to(device)

def entropy_node_dropout(data: Data, model, dropout_rate: float, noise_level: float):
    original_num_nodes = data.num_nodes
    original_num_edges = data.edge_index.shape[1]
    
    delta_entropy = kd_retention(model, data, noise_level)
    num_nodes = data.num_nodes
    num_drop = int(num_nodes * dropout_rate)
    
    # Sort nodes by delta entropy and get indices of nodes to drop
    drop_indices = np.argsort(delta_entropy)[-num_drop:]
    
    # Create a mask for nodes to keep
    keep_mask = torch.ones(num_nodes, dtype=torch.bool)
    keep_mask[drop_indices] = False
    
    # Apply the mask to node features and edge indices
    keep_mask = keep_mask.cpu()
    new_x = data.x[keep_mask]
    new_edge_index = data.edge_index[:, (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)]
    new_edge_index = new_edge_index.cpu()
    new_edge_index = new_edge_index[:, keep_mask[new_edge_index[0]] & keep_mask[new_edge_index[1]]]
    
    # Update node indices in edge_index
    node_map = torch.cumsum(keep_mask, dim=0) - 1
    new_edge_index = node_map[new_edge_index]
    
    # Adjust masks
    new_train_mask = data.train_mask[keep_mask]
    new_val_mask = data.val_mask[keep_mask]
    new_test_mask = data.test_mask[keep_mask]
    
    # Create a new Data object with dropped nodes
    new_data = Data(x=new_x, edge_index=new_edge_index, y=data.y[keep_mask],
                    train_mask=new_train_mask, val_mask=new_val_mask, test_mask=new_test_mask)
    
    # Verification checks
    nodes_dropped = original_num_nodes - new_data.num_nodes
    edges_dropped = original_num_edges - new_data.edge_index.shape[1]
    
    logging.info(f"Nodes dropped: {nodes_dropped} ({nodes_dropped/original_num_nodes:.2%})")
    #logging.info(f"Edges dropped: {edges_dropped} ({edges_dropped/original_num_edges:.2%})")
    
    if abs(nodes_dropped - num_drop) > 1:  # Allow for small rounding differences
        logging.warning(f"Unexpected number of nodes dropped. Expected: {num_drop}, Actual: {nodes_dropped}")
    
    if new_data.num_nodes >= original_num_nodes:
        logging.error("No nodes were dropped!")
        raise ValueError("Node dropping failed: no nodes were removed")
    
    return new_data.to(device)

set_seed(3164711608)
def train_and_get_results(data, model, optimizer, dropout_rate, noise_level,freqdrop):
    avg_testacc_before = []
    avg_acc_testallsplits_before = []
    avg_valacc_before = []
    avg_acc_valallsplits_before = []
    criterion = torch.nn.CrossEntropyLoss()

    def train(model, optimizer, data,epoch):
        model.train()
        optimizer.zero_grad()
        if epoch % freqdrop == 0:
            train_data = entropy_node_dropout(data, model, dropout_rate, noise_level)
        else:
            train_data = data
        #train_data = entropy_node_dropout(data, model, dropout_rate, noise_level)
        out = model(train_data.x, train_data.edge_index)
        loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        train_correct = pred[train_data.train_mask] == train_data.y[train_data.train_mask]
        train_acc = int(train_correct.sum()) / int(train_data.train_mask.sum())
        return loss, train_data.num_nodes, train_acc, train_data

    def val(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
        return val_acc

    def test(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc, pred, out

    for split_idx in range(0, 100):
        model.reset_parameters()
        optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
        
        # Ensure data has the correct mask structure
        if data.train_mask.dim() == 2:
            data.train_mask = data.train_mask[:, split_idx]
            data.val_mask = data.val_mask[:, split_idx]
            data.test_mask = data.test_mask[:, split_idx]

        # Check for data leakage
        train_nodes = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        test_nodes = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        val_nodes = data.val_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        
        if len(np.intersect1d(train_nodes, test_nodes)) > 0 or len(np.intersect1d(train_nodes, val_nodes)) > 0:
            logging.error("Data leakage detected. Stopping execution.")
            return

        if len(np.intersect1d(train_nodes, test_nodes)) > 0 or len(np.intersect1d(train_nodes, val_nodes)) > 0:
            logging.error("Data leakage detected. Stopping execution.")
            return





        logging.info(f"Training for index = {split_idx}")

        for epoch in range(1, 101):
            loss, num_nodes_after_dropout, train_acc, train_data = train(model, optimizer, data,epoch)
            if epoch % 10 == 0:  # Log every 10 epochs
                logging.info(f"Epoch {epoch}, Loss: {loss:.4f}, Nodes after dropout: {num_nodes_after_dropout}, Train Acc: {train_acc:.4f}")

        # Use the last train_data for validation and testing
        val_acc = val(model, data)
        avg_valacc_before.append(val_acc * 100)
        test_acc, pred, out = test(model, data)
        avg_testacc_before.append(test_acc * 100)
        logging.info(f'Val Accuracy: {val_acc*100:.4f}')
        logging.info(f'Test Accuracy: {test_acc*100:.4f}')
        logging.info("")
        avg_acc_valallsplits_before.append(np.mean(avg_valacc_before))
        avg_acc_testallsplits_before.append(np.mean(avg_testacc_before))

    visualize_prediction_confidence_and_entropy(
        model, train_data, pred, out.softmax(dim=1), train_data.train_mask, train_data.test_mask, split_idx, train_data.y, noise_level=1.0)
    
    return avg_acc_testallsplits_before, avg_acc_valallsplits_before