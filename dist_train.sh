nohup accelerate launch --main_process_port=29730 train_dior.py \
    --pretrained_model_name_or_path=stable-diffusion-v1-4 \
    --train_data_dir=path_to_data/DIOR/train \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=40 \
    --allow_tf32 \
    --checkpointing_steps=300 \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --max_grad_norm=1 \
    --lr_scheduler=constant --lr_warmup_steps=0 \
    --output_dir=checkpoint-dior >> train_dior.log 2>&1 &