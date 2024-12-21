echo "Running graphclassification..."
python graphclass.py \
    --dataset MUTAG \
    --model GraphConv \
    --num_layers 5 \
    --hidden_dimension 64 \
    --batch_size 32 \
    --num_epochs 500 \
    --out results.csv