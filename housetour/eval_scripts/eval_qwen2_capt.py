import json
from tqdm import tqdm

from qwen2 import qwen2_generate

if __name__ == "__main__":
    # Captioning Eval
    with open("/scratch/users/atacelen/Reconstructions3D/annotations_cleaned_v2.json", "r") as f:
        capt = json.load(f)

    with open("/scratch/users/atacelen/Reconstructions3D/val_indexes.txt", "r") as f:
        val_idxs = f.read()
        val_idxs = val_idxs.split("\n")
        val_idxs = [int(v) for v in val_idxs]

    capt = [d for d in capt if d["scene_id"] in val_idxs]

    preds = []

    for c in tqdm(capt):
        pred = qwen2_generate(c['scene_id'], c['candidates'], c['text']['instruction'])
        
        # print(f"Scene ID: {c['scene_id']}")
        # print(f"Pred: {pred}")
        # print("\n")

        preds.append(
            {
                "scene_id" : c['scene_id'], 
                "instruction" : c['text']['instruction'],
                "predicted_answer" : pred,
                "ground_truth_answer" : c['text']['response']
            }
        )
    
    output_file = "eval_qwen2_capt.jsonl"
    with open(output_file, "w") as file:
        for item in preds:
            json.dump(item, file)
            file.write("\n")  # Add a newline after each JSON object