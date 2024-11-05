# instructions
# use llava med env commands in llava_med_instructions.py
# pip install -e . inside Llava folder
# redo the install transformer from llava med instructions
# pip install 'accelerate>=0.26.0'
# pip install -e ".[train]"
# pip install flash-attn --no-build-isolation

# pip install deepspeed // may not be necessary
# pip install torch --upgrade // maybe not necessary

# THEN EDIT FILES if not already done
# /home/malindal/LLaVA-Med/LLaVA/llava/train/train.py line 850 and 911 changed model loading to load from llavamed mistral
# /home/malindal/miniconda3/envs/llava/lib/python3.10/site-packages/deepspeed/launcher/launch.py line 165 hard coded cuda visible device = 1
#     # current_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, local_gpu_ids))
#     current_env["CUDA_VISIBLE_DEVICES"] = "1"

#/home/malindal/LLaVA-Med/llava/model/builder.py         kwargs['torch_dtype'] = torch.bfloat16 and also lines 63, 88-91
# /home/malindal/LLaVA-Med/llava/model/llava_arch.py          image_features = self.get_model().get_vision_tower()(images).to(torch.bfloat16) # changed to bfloat16
#         self.get_model().mm_projector = self.get_model().mm_projector.to(torch.bfloat16) # changed to bfloat16


deepspeed  --num_gpus=1 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path ./llava-med-v1.5-mistral-7b \
    --data_path /home/malindal/high_modality/ultrasound/busi/annotation_train.jsonl \
    --image_folder ../../../../high_modality/ultrasound/busi \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/ultrasound/busi/finetune_version1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# things to note:
# default hyperparameters:
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
# num workers 4
# learning rate 2e-4

# in same env and folder
python scripts/merge_lora_weights.py --model-path /home/malindal/LLaVA-Med/LLaVA/checkpoints/ultrasound/busi/finetune_version1 \
     --model-base  microsoft/llava-med-v1.5-mistral-7b \
     --save-model-path ./checkpoints/ultrasound/busi/merged_version1


# now in regular llava med env
PYTHONPATH=. python llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path ./LLaVA/checkpoints/chest_xray/vindr/merged_version2 \
    --question-file ../high_modality/chest_xray/vindr/annotation_test.jsonl \
    --image-folder ../high_modality/chest_xray/vindr/ \
    --answers-file ../high_modality/chest_xray/vindr/llava_med_test_results_finetuned2.jsonl \
    --temperature 0.0
# note model path, asnwers file


# preliminary results
# 1: 1, 1, 8, 2 workers, cin
# 2: 16, 4, 1, 2 workers, nev nev nev nev
# 3: 16, 4, 1, 4 workers, learning rate * 2, Nevus, sometimes results that were similar but not in the multiple choice list

# scratch
# from transformers import AutoModel, AutoTokenizer
from llava.model.builder import load_pretrained_model

# model_name = "microsoft/llava-med-v1.5-mistral-7b"
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer, model, image_processor, context_len = load_pretrained_model("microsoft/llava-med-v1.5-mistral-7b", None, "llava-med-v1.5-mistral-7b")

# /home/malindal/high_modality/derm/ham10000/annotation_train.jsonl
# ../../../../high_modality/derm/ham10000/annotation_train.jsonl
#!/bin/bash


