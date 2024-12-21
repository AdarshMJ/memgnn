import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
import numpy as np
import time
import csv
from model import *
from dataloader import *
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from graphclassentropy import *

parser = argparse.ArgumentParser(description='Run Graph Classification script')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
parser.add_argument('--model', type=str, default='GraphConv', choices=['GATv2','SimpleGCN','SimpleGCNRes','GraphConv'], help='Model to use')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--device',type=str,default='cuda',help='Device to use')
parser.add_argument('--num_epochs',type=int,default='100',help='Number of training epochs')
parser.add_argument('--batch_size',type=int,default='32',help='Batch size for training')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
filename = args.out
p = args.dropout
hidden_dimension = args.hidden_dimension
lr = args.lr

print("Loading dataset...")

if args.dataset in ['MUTAG','ENZYMES','PROTEINS']:
    dataset = TUDataset(root='./data/TUDataset', name=args.dataset, transform=NormalizeFeatures())
    
    # Split dataset into train, validation, and test
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    num_features = dataset.num_features
    num_classes = dataset.num_classes
else:
    print("Invalid dataset")
    sys.exit()

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Training graphs: {len(train_dataset)}')
print(f'Validation graphs: {len(val_dataset)}')
print(f'Test graphs: {len(test_dataset)}')

print()
print("Start Training...")

def create_model(model, num_features, num_classes, hidden_dimension):
    if model == 'GATv2':
        return GATv2(num_features, hidden_dimension, num_classes, args.num_layers)
    elif model == 'SimpleGCN':
        return SimpleGCN(num_features, num_classes, hidden_dimension, args.num_layers)
    elif model == 'SimpleGCNRes':
        return SimpleGCNRes(num_features, num_classes, hidden_dimension, args.num_layers)
    elif model == 'GraphConv':
        return GraphConv(num_features=num_features, 
                        num_classes=num_classes,
                        hidden_channels=hidden_dimension,
                        num_layers=args.num_layers)
    else:
        print("Invalid Model")
        sys.exit()

def create_model_instance():
    return create_model(args.model, num_features, num_classes, args.hidden_dimension).to(device)

def create_optimizer_params():
    return {
        'lr': args.lr,
        'weight_decay': 0.0
    }

print(f"Model Architecture: {args.model}")
model = create_model_instance()
print(f"Model Details:\n{model}")

gcn_start = time.time()

# Run training with multiple seeds
seeds = [3164711608, 894959334, 2487307261, 3349051410, 493067366]
final_train_accs, final_test_accs = run_multiple_seeds(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model_class=create_model_instance,
    optimizer_class=torch.optim.Adam,
    optimizer_params=create_optimizer_params(),
    num_epochs=args.num_epochs,
    dataset_name=args.dataset,
    num_layers=args.num_layers,
    seeds=seeds
)

gcn_end = time.time()

# Calculate mean and standard deviation
mean_train_acc = np.mean(final_train_accs)
std_train_acc = np.std(final_train_accs)
mean_test_acc = np.mean(final_test_accs)
std_test_acc = np.std(final_test_accs)

print(f"Final Train Accuracy: {mean_train_acc:.2f} ± {std_train_acc:.2f}")
print(f"Final Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")

# Write results to CSV
headers = ['Method','Model','Dataset','AvgTrainAcc','DevTrain','AvgTestAcc','DevTest',
          'HiddenDim','LR','Dropout']

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    if file.tell() == 0:
        writer.writerow(headers)
    writer.writerow(['GraphClassification', args.model, args.dataset,
                    f"{final_train_accs[-1]:.4f}", f"{std_test_acc:.4f}", 
                    f"{mean_test_acc:.4f}", f"{std_test_acc:.4f}",
                    hidden_dimension, lr, p
    ])