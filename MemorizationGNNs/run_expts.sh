# echo "Running baseline..."
# python baselinegcn.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 32 \
#   --num_epochs 1000

# echo "Running random dropping..."
# python checkmem.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 32 \
#   --percentile 2 \
#   --num_epochs 1000



echo "Running static dropping..."
python checkmem.py \
  --dataset Cora \
  --out changeval.csv \
  --model SimpleGCN\
  --lr 0.01 \
  --hidden_dimension 32 \
  --percentile 98 \
  --num_epochs 1000


# echo "Running static dropping..."
# python checkmem.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 32 \
#   --percentile 50 \
#   --num_epochs 200




# echo "Running static dropping..."
# python checkmem.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 32 \
#   --percentile 80

# echo "Running static dropping..."
# python checkmem.py \
#   --dataset Cora \
#   --out changeval.csv \
#   --model SimpleGCN\
#   --lr 0.01 \
#   --hidden_dimension 32 \
#   --percentile 80