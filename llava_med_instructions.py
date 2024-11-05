# don't run this file directly, just follow the instructions:

# cd LLaVA-Med
# tmux
# conda create -n llava-med python=3.10 -y
# conda activate llava-med
# pip install -e .

# pip install -U transformers
# pip install protobuf==3.20.0

# add cache_position=None to llava/model/language_model/llava_mistral.py if not already added
# edit llava/eval/model_vqa.py to match our jsonl format if not already edited (see colab)

# PYTHONPATH=. python llava/eval/model_vqa.py \
#     --conv-mode mistral_instruct \
#     --model-path microsoft/llava-med-v1.5-mistral-7b \
#     --question-file ../high_modality/chest_xray/mimic-cxr/annotation_test.jsonl \
#     --image-folder ../high_modality/chest_xray/mimic-cxr/ \
#     --answers-file ../high_modality/chest_xray/mimic-cxr/llava_med_test_results.jsonl \
#     --temperature 0.0

PYTHONPATH=. python llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path microsoft/llava-med-v1.5-mistral-7b \
    --question-file ../high_modality/chest_xray/vindr/annotation_test.jsonl \
    --image-folder ../high_modality/chest_xray/vindr/ \
    --answers-file ../high_modality/chest_xray/vindr/llava_med_test_results.jsonl \
    --temperature 0.0

# full paath
PYTHONPATH=. python /home/malindal/LLaVA-Med/llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path /home/malindal/LLaVA-Med/LLaVA/checkpoints/merged_version1 \
    --question-file /home/malindal/high_modality/derm/ham10000/annotation_valid.jsonl \
    --image-folder /home/malindal/high_modality/derm/ham10000/ \
    --answers-file /home/malindal/high_modality/derm/ham10000/llava_med_test_results_finetuned.jsonl \
    --temperature 0.0

# from LLava
PYTHONPATH=. python ../llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path ./checkpoints/merged_version1 \
    --question-file ../../high_modality/derm/ham10000/annotation_valid.jsonl \
    --image-folder ../../high_modality/derm/ham10000/ \
    --answers-file ../../high_modality/derm/ham10000/llava_med_test_results_finetuned.jsonl \
    --temperature 0.0

# from LLavaMed
PYTHONPATH=. python llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path ./LLaVA/checkpoints/merged_version2 \
    --question-file ../high_modality/derm/ham10000/annotation_valid.jsonl \
    --image-folder ../high_modality/derm/ham10000/ \
    --answers-file ../high_modality/derm/ham10000/llava_med_test_results_finetuned2.jsonl \
    --temperature 0.0

# PYTHONPATH=. python llava/eval/model_vqa.py \
#     --conv-mode mistral_instruct \
#     --model-path ./LLaVA/checkpoints/finetune_version1 \
#     --question-file ../high_modality/derm/ham10000/annotation_valid.jsonl \
#     --image-folder ../high_modality/derm/ham10000/ \
#     --answers-file ../high_modality/derm/ham10000/llava_med_test_results_finetuned.jsonl \
#     --temperature 0.0

#tmux attach -d -t ssh-to-s3

