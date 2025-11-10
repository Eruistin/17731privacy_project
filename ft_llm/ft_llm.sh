python ft_llm.py \
  --data_dir ../data/shadow/wiki_json --train_file "train_finetune.json"\
  -m gpt2 --block_size 512 --epochs 15 --batch_size 8 --gradient_accumulation_steps 1 \
  --lr 2e-4 --outdir ../models/shadow/gpt2_3_lora32_adamw_b8_lr2 \
  --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05

# ft_llm_colab.py ...