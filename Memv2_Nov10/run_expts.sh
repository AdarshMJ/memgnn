#3164711608,894959334,2487307261,3349051410,493067366
# echo "Running baseline..."
# python baselinegcn.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 32 \
#   --num_epochs 1500 \
#   --num_layers 5 \


echo "Running GATv2 baseline..."
python baselinegcn.py \
  --dataset Cora \
  --out changeval.csv \
  --model GATv2 \
  --lr 0.01 \
  --hidden_dimension 64 \
  --num_epochs 1200 \
  --num_layers 5 \

# echo "Running baseline..."
# python baselinegcn.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 64 \
#   --num_epochs 1200 \
#   --num_layers 5 \
#   --seed 894959334

# echo "Running baseline..."
# python baselinegcn.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 64 \
#   --num_epochs 1200 \
#   --num_layers 5 \
#   --seed 2487307261

# echo "Running baseline..."
# python baselinegcn.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 64 \
#   --num_epochs 1200 \
#   --num_layers 5 \
#   --seed 3349051410



# echo "Running baseline..."
# python baselinegcn.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 64 \
#   --num_epochs 1200 \
#   --num_layers 5 \
#   --seed 493067366


# echo "Running baseline..."
# python baselinegcn.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model GATv2\
#   --lr 0.01 \
#   --hidden_dimension 64 \
#   --num_epochs 1200 \
#   --num_layers 5 \
#   --seed 3164711608