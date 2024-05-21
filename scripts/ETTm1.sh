export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720

do
    python -u run.py \
  --is_training 1 \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --model AlignTime \
  --train_epochs 10 \
  --patience 3
done



