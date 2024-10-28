echo "Running baseline..."
python baselinegcn.py \
  --dataset Cora \
  --out changeval.csv \
  --model SimpleGCN\
  --lr 0.001 \
  --hidden_dimension 512 \
  --num_train 20 \
  --num_val 500 \