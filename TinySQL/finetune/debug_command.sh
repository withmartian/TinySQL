# num_train_epochs=3
# batch_size=8
# gradient_accumulation_steps=1
# warmup_steps=50
# max_seq_length=512
# weight_decay=0.01


python finetune.py --finetune \
    --model_name "roneneldan/TinyStories-Instruct-2Layers-33M" \
    --learning_rate "1e-5" \
    --warmup_steps 50 \
    --num_train_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512 \
    --weight_decay 0.01 \
    --dataset_name "withmartian/cs11_dataset" \
    --max_seq_length 512

python finetune.py --finetune \
    --model_name "roneneldan/TinyStories-Instruct-2Layers-33M" \
    --learning_rate "2e-5" \
    --warmup_steps 50 \
    --num_train_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512 \
    --weight_decay 0.01 \
    --dataset_name "withmartian/cs11_valid" \
    --max_seq_length 256

python finetune.py --finetune \
    --model_name "roneneldan/TinyStories-Instruct-2Layers-33M" \
    --learning_rate "2e-5" \
    --warmup_steps 50 \
    --num_train_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512 \
    --weight_decay 0.01 \
    --dataset_name "withmartian/cs12_dataset" \
    --max_seq_length 256

python finetune.py --finetune \
    --model_name "roneneldan/TinyStories-Instruct-2Layers-33M" \
    --learning_rate "2e-5" \
    --warmup_steps 50 \
    --num_train_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512 \
    --weight_decay 0.01 \
    --dataset_name "withmartian/cs12_valid" \
    --max_seq_length 256