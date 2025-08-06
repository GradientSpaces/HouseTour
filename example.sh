# Run Qwen2-VL-3D Training
python train_qwen2_vl_3d.py \
    --lora_adapter_dir "/scratch/users/atacelen/official-code/training_checkpoints/lora_adapter/checkpoint-0" \
    --data_dir "/scratch/users/atacelen/house_tour_dataset/Reconstructions3D" \
    --diffuser_dir "/scratch/users/atacelen/official-code/checkpoints/residual-diffuser" \
    --traj_data "/scratch/users/atacelen/diffuser/trajectories.jsonl"

# Run Qwen2-VL Supervised Fine-Tuning (SFT)
python train_qwen2_vl.py \
    --data_dir "/scratch/users/atacelen/house_tour_dataset/Reconstructions3D" \
    --output-dir "training_checkpoints/lora_adapter" 

# Run Qwen2-VL-3D Evaluation 
python eval_qwen2_vl_3d.py \
    --model-path "/scratch/users/atacelen/official-code/checkpoints/qwen2-vl-3d" \
    --traj-path "/scratch/users/atacelen/official-code/checkpoints/residual-diffuser" \
    --traj-data "/scratch/users/atacelen/diffuser/trajectories.jsonl"

# Run Residual Diffuser Training
python -m scripts.train_residual_diffuser

# Run Residual Diffuser Testing
python -m scripts.test_residual_diffuser

