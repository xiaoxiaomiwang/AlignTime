export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720

do
    python -u run.py \
  --is_training 1\
  --data_path exchange_rate.csv \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --enc_in 8 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --model AlignTime \
  --d_model 512 \
  --train_epochs 10 \
  --patience 3
done







