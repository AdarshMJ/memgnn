import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import time
import csv
from model import *
from trainmem import *
from dataloader import *

def parse_args():
    parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
    parser.add_argument('--dataset', type=str, help='Dataset to download')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
    parser.add_argument('--model', type=str, default='SimpleGCN', choices=['GCN', 'GATv2','SimpleGCN','SGC','MLP'], help='Model to use')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
    parser.add_argument('--out', type=str, help='name of log file')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
    parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
    parser.add_argument('--device',type=str,default='cuda',help='Device to use')
    parser.add_argument('--num_train',type=int,default='20',help='Number of training nodes per class')
    parser.add_argument('--num_val',type=int,default='500',help='Number of validation nodes')
    parser.add_argument('--num_epochs',type=int,default='100',help='Number of training epochs')
    #parser.add_argument('--percentile',type=int,default='None',help='Number of nodes to drop')
    parser.add_argument('--percentile', type=str, default=None)
    args = parser.parse_args()
    
    # Convert string 'None' to actual None
    if args.percentile == 'None':
        args.percentile = None
    else:
        args.percentile = float(args.percentile)
    
    return args

args = parse_args()




filename = args.out
p = args.dropout
hidden_dimension = args.hidden_dimension
lr = args.lr




print("Loading dataset...")

if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo','DBLP','penn94','reed98']:
    data, num_classes,num_features,num_train_nodes,num_test_nodes,num_val_nodes = load_data(args.dataset,args.num_train,args.num_val)


elif args.dataset in ['Roman-empire','Minesweeper','Amazon-ratings','Questions']:
    data, num_classes,num_features = load_data(args.dataset)

elif args.dataset in ['cornell.npz','texas.npz','wisconsin.npz']:
    path = '/home/adarshjamadandi/heterophilous-graphs/data/'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print()
    print("Splitting datasets train/val/test...")
    transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_test=0.3,num_val=0.1)
    data  = transform2(data)
    data = data.to(device)
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")

elif args.dataset in ['chameleon_filtered.npz','squirrel_filtered.npz','actor.npz']:
    path = '/home/adarshjamadandi/heterophilous-graphs/data/'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print("Splitting datasets train/val/test...")
    # transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_train_per_class = 10,num_val=20)
    # #transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_test=0.2,num_val=0.2)
    # data  = transform2(data)
    transform2 = RandomNodeSplit(split="test_rest", num_splits=100, num_train_per_class=20, num_val=500)
    data = transform2(data)
    data.train_mask = data.train_mask | data.val_mask
    data.val_mask = torch.zeros_like(data.val_mask)  # Clear validation mask
    num_train_nodes = data.train_mask.sum().item()
    num_val_nodes = data.val_mask.sum().item()
    num_test_nodes = data.test_mask.sum().item()
    print()    
    data = data.to(device)
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")

else :
    print("Invalid dataset")
    sys.exit()



print()
print(f"Number of training nodes: {num_train_nodes/100}")
print(f"Number of validation nodes: {num_val_nodes/100}")
print(f"Number of test nodes: {num_test_nodes/100}")
print()

print("Start Training...")


def create_model(model, num_features, num_classes, hidden_dimension):
    if model == 'GCN':
        return GCN(num_features, num_classes, hidden_dimension, num_layers=args.num_layers)
    elif model == 'GATv2':
        return GATv2(num_features, 8, num_classes)
    elif model == 'SimpleGCN':
        return SimpleGCN(num_features, num_classes, hidden_dimension)
    elif model == 'SGC':
        return SGC(num_features, num_classes)
    else:
        print("Invalid Model")
        sys.exit()


model = create_model(args.model, num_features, num_classes, args.hidden_dimension).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0)
model = model.to(device)
print(model)



gcn_start = time.time()
#finaltestaccafter,finalvalaccafter = train_and_get_results(data, model,optimizer,p)
finaltestaccafter,finalvalaccafter,finaltrainaccafter   = train_and_get_results(data, model, optimizer, args.num_epochs,percentile=args.percentile,k_nodes=None,dynamic_zeroing=False)
#finaltestaccafter,finalvalaccafter,finaltrainaccafter  = train_and_get_results(data, model, optimizer,args.num_epochs,percentile=args.percentile, k_nodes=5,dynamic_zeroing=True,random_baseline=False)
gcn_end = time.time()


avg_test_acc_after = np.mean(finaltestaccafter)
sample_size = len(finaltestaccafter)
std_dev_after = 2 * np.std(finaltestaccafter)/(np.sqrt(sample_size))

avg_val_acc_after = np.mean(finalvalaccafter)
sample_size = len(finalvalaccafter)
std_devval_after = 2 * np.std(finalvalaccafter)/(np.sqrt(sample_size))

avg_train_acc_after = np.mean(finaltrainaccafter)
sample_size = len(finaltrainaccafter)
std_devtrain_after = 2 * np.std(finaltrainaccafter)/(np.sqrt(sample_size))

#print(f'Final val accuracy  {(avg_val_acc_after):.4f}\u00B1{(std_devval_after):.4f}')
#print(f'Final test accuracy {(avg_test_acc_after):.4f}\u00B1{(std_dev_after):.4f}')

headers = ['Method','Model','Dataset','AvgTrainAcc','TrainDeviation','AvgTestAccAfter','DeviationAfter',
        'HiddenDim','LR','Dropout','TrainNodes','TestNodes','ValNodes','GCNTime']

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    if file.tell() == 0:
        writer.writerow(headers)
    writer.writerow([f'NodeDrop_{args.percentile}',args.model, args.dataset,f"{(avg_train_acc_after):.4f}", f"{(std_devtrain_after):.4f}", f"{(avg_test_acc_after):.4f}", f"{(std_dev_after):.4f}",
                     hidden_dimension, lr, p, num_train_nodes/100,num_test_nodes/100,num_val_nodes/100, gcn_end - gcn_start])
