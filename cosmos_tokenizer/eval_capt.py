import json
from tqdm import tqdm
import torch
import yaml
import os

from cosmos_tokenizer.models.model import CosmosLlama
from cosmos_tokenizer.conversation.chat import Chat, conv_llava_llama_2
from cosmos_tokenizer.processors.img_array_processors import ImgArrayProcessor

path_to_recon = "/scratch/users/atacelen/Reconstructions3D"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = yaml.safe_load(open("cosmos_tokenizer/configs/cosmos_llama_eval.yaml"))
model = CosmosLlama(cfg).to(device)

ckpt_path = "results/train_cosmos_capt_e2_loss1.329.pth"
model.load_state_dict(torch.load(ckpt_path))
model.eval()

vis_processor = ImgArrayProcessor.from_config(cfg["preprocess"]["vis_processor"]["eval"])

def cosmos_tokenizer_generate(scene_id, candidates, instruction):
    chat = Chat(model, vis_processor, device)
    chat_state = conv_llava_llama_2.copy()
    chat_state.system = "" 

    img_array_path = [os.path.join(path_to_recon, f"{scene_id}_video", "keyframes_resized", c) for c in candidates]
    img_array = vis_processor(img_array_path)

    img_array = img_array.unsqueeze(0).to(device)
    img_array_embed, _ = model.encode_cosmos_visual(img_array)
    img_list = []
    img_list.append(img_array_embed)

    chat_state.append_message(chat_state.roles[0], "<Video><ImageHere></Video> ")

    user_message = instruction
    chat.ask(user_message, chat_state)

    # Get an answer
    llm_answer, tokens, transition_scores, attn = chat.answer(
        conv = chat_state,
        img_list = img_list,
        num_beams = 1,
        temperature = 0.3,
        max_new_tokens = 1000,
        max_length = 2000,
    )

    return llm_answer


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

        pred = cosmos_tokenizer_generate(c['scene_id'], c['candidates'], c['text']['instruction'])
        
        print(f"Scene ID: {c['scene_id']}")
        print(f"Pred: {pred}")
        print("\n")

        preds.append(
            {
                "scene_id" : c['scene_id'], 
                "instruction" : c['text']['instruction'],
                "predicted_answer" : pred,
                "ground_truth_answer" : c['text']['response']
            }
        )
    
    output_file = "eval_cosmos_capt.jsonl"
    with open(output_file, "w") as file:
        for item in preds:
            json.dump(item, file)
            file.write("\n")  # Add a newline after each JSON object