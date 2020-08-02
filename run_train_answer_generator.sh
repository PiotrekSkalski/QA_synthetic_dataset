python train_answer_generator.py \
    --model_path bert-large-uncased \
    --output_dir answer_generator_bert-large-uncased_2 \
    --data_dir data \
    --max_seq_length 512 \
    --max_ans_length 64 \
    --do_lower_case \
    --num_train_epochs 2 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --dev_batch_size 8 \
    --dynamic_batching \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --train_logging_steps 10 \
    --dev_logging_steps 250 \
    --save_steps 1000 \
    --overwrite_output_dir \
    --evaluate_during_training \
    --threads 2 \
    --fp16 \
    --fp16_opt_level O2 \
