export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720

do
  python -u run.py \
  --is_training 1 \
  --data_path weather.csv \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --enc_in 21 \
  --itr 1 \
  --model AlignTime \
  --d_model 1024 \
  --train_epochs 10 \
  --patience 3
done

