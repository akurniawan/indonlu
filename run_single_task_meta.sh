# mBERT-based model
MODEL1="mbert-multi-emb-10_embeddings+layer_norm_emb"
MODEL2="mbert-multi-emb-10_embeddings+layer_norm_emb+layer.0.+layer.1.+layer.2.+layer.3."
MODEL3="mbert-multi-emb-10_embeddings+layer_norm_emb+layer.0.+layer.1.+layer.2.+layer.3.+layer.4.+layer.5."
MODEL4="mbert-multi-emb-10_embeddings+layer_norm_emb+layer.0.+layer.1.+layer.2.+layer.3.+layer.4.+layer.5.+layer.6."
# embedding
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint $MODEL1 --step_size 1 --gamma 0.9 --experiment_name $MODEL1_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
# embedding+layer1
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint $MODEL2 --step_size 1 --gamma 0.9 --experiment_name $MODEL2_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
# embedding+layer1+layer2
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint $MODEL3 --step_size 1 --gamma 0.9 --experiment_name $MODEL3_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
# embedding+layer1+layer2+layer3
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint $MODEL4 --step_size 1 --gamma 0.9 --experiment_name $MODEL4_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force