import json
from tqdm import tqdm

from qwen2 import qwen2_generate

if __name__ == "__main__":
    # Captioning Eval
    with open("/scratch/users/atacelen/Reconstructions3D/annotations_true_false.json", "r") as f:
        tf = json.load(f)

    with open("/scratch/users/atacelen/Reconstructions3D/val_indexes.txt", "r") as f:
        val_idxs = f.read()
        val_idxs = val_idxs.split("\n")
        val_idxs = [int(v) for v in val_idxs]

    tf = [d for d in tf if d["scene_id"] in val_idxs]

    correct, incorrect = 0, 0
    preds = []

    for c in tqdm(tf):
        print(f"Scene ID: {c['scene_id']}")
        print(f"Question: {c['text']['instruction']}")
        print(f"GT: {c['text']['response']}")
        
        pred = qwen2_generate(c['scene_id'], c['candidates'], c['text']['instruction'])
        preds.append(
            {
                "scene_id" : c['scene_id'], 
                "instruction" : c['text']['instruction'],
                "predicted_answer" : pred,
                "ground_truth_answer" : c['text']['response']
            }
        )
        
        print(f"Pred: {pred}")
        if pred == c['text']['response']:
            print("Correct!")
            correct += 1
        else:
            print("Incorrect")
            incorrect += 1
        
        print(f"Rolling Accuracy: {correct / (correct + incorrect)}")
    
    output_file = "eval_qwen2_tf.jsonl"
    with open(output_file, "w") as file:
        for item in preds:
            json.dump(item, file)
            file.write("\n")  # Add a newline after each JSON object