# XLM-based model

# embedding
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force
# embedding+layer1
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force
# embedding+layer1+layer2
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force
# embedding+layer1+layer2+layer3
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force
# embedding+layer1+layer2+layer3+layer4
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force
# embedding+layer1+layer2+layer3+layer4+layer5
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force
# embedding+layer1+layer2+layer3+layer4+layer5+layer6
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force
# embedding+layer1+layer2+layer3+layer4+layer5+layer6+layer7
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force
# embedding+layer1+layer2+layer3+layer4+layer5+layer6+layer8
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint xlm-mlm-100-1280 --step_size 1 --gamma 0.9 --experiment_name xlm-mlm-100-1280_b$4_step1_gamma0.9_lr1e-5_early$3_layer16_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 16 --force

# mBERT-based model
# embedding
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint bert-base-multilingual-uncased --step_size 1 --gamma 0.9 --experiment_name bert-base-multilingual-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
# embedding+layer1
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint bert-base-multilingual-uncased --step_size 1 --gamma 0.9 --experiment_name bert-base-multilingual-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
# embedding+layer1+layer2
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint bert-base-multilingual-uncased --step_size 1 --gamma 0.9 --experiment_name bert-base-multilingual-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
# embedding+layer1+layer2+layer3
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint bert-base-multilingual-uncased --step_size 1 --gamma 0.9 --experiment_name bert-base-multilingual-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
# embedding+layer1+layer2+layer3+layer4
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint bert-base-multilingual-uncased --step_size 1 --gamma 0.9 --experiment_name bert-base-multilingual-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force
# embedding+layer1+layer2+layer3+layer4+layer5+layer6
CUDA_VISIBLE_DEVICES=$1 python3 main.py --n_epochs 1 --train_batch_size $4 --model_checkpoint bert-base-multilingual-uncased --step_size 1 --gamma 0.9 --experiment_name bert-base-multilingual-uncased_b$4_step1_gamma0.9_lr1e-5_early$3_layer12_lowerFalse --lr 1e-5 --early_stop $3 --dataset $2 --num_layers 12 --force