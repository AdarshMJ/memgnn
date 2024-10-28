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



# def visualize_prediction_confidence_and_entropy(model, data, predictions, confidences, test_mask, split_idx, true_labels, noise_level=0.1):
#     # Filter predictions for test nodes
#     test_predictions = predictions[test_mask]
#     test_confidences = confidences[test_mask]
#     true_labels = data.y[test_mask]
#     test_mask = test_mask.cpu()

#     # Calculate KD retention (assuming this function is defined elsewhere)
#     delta_entropy = kd_retention(model, data, noise_level)
#     test_delta_entropy = delta_entropy[test_mask]

#     # Determine high delta entropy nodes (e.g., top 10%)
#     high_entropy_threshold = np.percentile(test_delta_entropy, 90)
#     high_entropy_mask = test_delta_entropy >= high_entropy_threshold

#     # Plotting
#     fig, ax = plt.subplots(figsize=(12, 10))

#     # Get max confidence for each prediction
#     max_confidences = torch.max(test_confidences, dim=1).values.detach().cpu().numpy()

#     # Determine correctness of predictions
#     correct_predictions = test_predictions == true_labels

#     # Create a colormap for the two possible cases
#     colormap = {
#         True: 'green',   # Correct prediction
#         False: 'red'     # Wrong prediction
#     }
#     colors = [colormap[cp.item()] for cp in correct_predictions]

#     # Plot low entropy nodes
#     low_entropy_scatter = ax.scatter(
#         max_confidences[~high_entropy_mask],
#         test_delta_entropy[~high_entropy_mask],
#         c=[c for c, he in zip(colors, high_entropy_mask) if not he],
#         alpha=0.6,
#         marker='o'
#     )

#     # Plot high entropy nodes with a different marker
#     high_entropy_scatter = ax.scatter(
#         max_confidences[high_entropy_mask],
#         test_delta_entropy[high_entropy_mask],
#         c=[c for c, he in zip(colors, high_entropy_mask) if he],
#         alpha=0.6,
#         marker='*',
#         s=100  # Larger size for visibility
#     )

#     ax.set_title(f'Confidence vs Delta Entropy Plot (Split {split_idx})')
#     ax.set_xlabel('Model Confidence')
#     ax.set_ylabel('Delta Entropy')

#     # Count nodes in each category
#     category_counts = {
#         'green': sum(1 for c in colors if c == 'green'),
#         'red': sum(1 for c in colors if c == 'red')
#     }

#     # Count high entropy nodes in each category
#     high_entropy_counts = {
#         'green': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'green' and he),
#         'red': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'red' and he)
#     }

#     # Create a custom legend with counts
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Correct Predictions: {category_counts["green"]} (High Δ Entropy: {high_entropy_counts["green"]})', markerfacecolor='green', markersize=10),
#         plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong Predictions: {category_counts["red"]} (High Δ Entropy: {high_entropy_counts["red"]})', markerfacecolor='red', markersize=10),
#         plt.Line2D([0], [0], marker='*', color='w', label=f'High Δ Entropy: {sum(high_entropy_mask)}', markerfacecolor='gray', markersize=15),
#     ]
#     ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

#     plt.tight_layout()
#     plt.savefig(f'confidence_vs_entropy_split_{split_idx}.png', bbox_inches='tight')
#     plt.close()

#     # Print the counts
#     print(f"Split {split_idx} Node Counts:")
#     print(f"Correct Predictions: {category_counts['green']} (High Δ Entropy: {high_entropy_counts['green']})")
#     print(f"Wrong Predictions: {category_counts['red']} (High Δ Entropy: {high_entropy_counts['red']})")
#     print(f"Total High Δ Entropy: {sum(high_entropy_mask)}")
#     print()

#     return category_counts, high_entropy_counts

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
    
set_seed(3164711608)


def train_and_get_results(data, model,optimizer,p):
    avg_testacc_before = []
    avg_acc_testallsplits_before = []
    avg_valacc_before = []
    avg_acc_valallsplits_before = []
    criterion = torch.nn.CrossEntropyLoss()

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
 
    for split_idx in range(0,100):
        model.reset_parameters()
        
        # Reset optimizer
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
            print(f"Warning: Found {len(leakage_nodes)} nodes in both the training and test sets. Stopping execution.")
            sys.exit(1)  # Exit the script due to data leakage
        if len(np.intersect1d(train_nodes, val_nodes)) > 0:
            print(f"Warning: Found {len(leakage_nodes)} nodes in both the training and validation sets. Stopping execution.")   
            sys.exit(1)  # Exit the script due to data leakage
        
        #=========================================================#
        
        
        print(f"Training for index = {split_idx}")

        for epoch in tqdm(range(1, 101)):
            loss = train(model,optimizer)
            #print(f"Epoch {epoch}, Loss: {loss:.4f}")  # Print training loss
        val_acc = val(model)  
        avg_valacc_before.append(val_acc*100)  
        test_acc,pred,out = test(model)    
        avg_testacc_before.append(test_acc*100)   
        print(f'Val Accuracy: {val_acc*100:.4f}') 
        print(f'Test Accuracy: {test_acc*100:.4f}')
        print()        
        avg_acc_valallsplits_before.append(np.mean(avg_valacc_before))  
        avg_acc_testallsplits_before.append(np.mean(avg_testacc_before))            
    
    #visualize_prediction_confidence_and_entropy(model, data, pred, out.softmax(dim=1), test_mask, split_idx, data.y, noise_level=1.0)
    visualize_prediction_confidence_and_entropy(
    model, data, pred, out.softmax(dim=1), train_mask, test_mask, split_idx, data.y, noise_level=1.0)
    return avg_acc_testallsplits_before,avg_acc_valallsplits_before

