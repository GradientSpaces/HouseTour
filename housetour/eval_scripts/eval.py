import yaml
import os
from PIL import Image
import torch
from torchvision import transforms
import einops
from torch.amp import autocast as autocast
import json
import numpy as np

from grandtour.models.grandtour import GrandTour
from grandtour.models.blip2_llama import Blip2Llama
from grandtour.conversation.chat import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2
from grandtour.processors.img_array_processors import ImgArrayProcessor

from attn_vis import heterogenous_stack, aggregate_attention

anno_path = "/scratch/users/atacelen/Reconstructions3D/annotations.json"
with open(anno_path, "r") as f:
    anno = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = yaml.safe_load(open("grandtour/configs/grandtour_eval.yaml"))
# cfg = yaml.safe_load(open("grandtour/configs/timechat.yaml"))
model = GrandTour.from_config(cfg["model"]).to(device)
# model = Blip2Llama.from_config(cfg["model"]).to(device)

ckpt_path = "results/model_ws_4_r16_a16_e10_cleaned.pth"
model.load_state_dict(torch.load(ckpt_path))
model.eval()

vis_processor = ImgArrayProcessor.from_config(cfg["preprocess"]["vis_processor"]["eval"])

chat = Chat(model, vis_processor, device)
print("Chat initialized")

# Upload the image
chat_state = conv_llava_llama_2.copy()
# chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
chat_state.system = ""
video_idx = 0
# anno_idx = next((item for item in anno if item.get("scene_id") == video_idx), None)
# candidates = anno_idx["candidates"]
# Optionally: Take the candidates from the candidates.txt
with open(f"/scratch/users/atacelen/Reconstructions3D/{video_idx}_video/candidates.txt", "r") as f:
    candidates = f.read().split("\n")
    candidates = list(sorted(candidates))

# Prepare the image array from candidates
img_list = []
keyframe_dir = f"/scratch/users/atacelen/Reconstructions3D/{video_idx}_video/keyframes_resized"

img_array_path = [os.path.join(keyframe_dir, c) for c in candidates]
img_array = vis_processor(img_array_path)

# Add the array to the chat
img_array = img_array.unsqueeze(0).to(device)
img_array_embed, _ = model.encode_videoQformer_visual(img_array)
# img_array_embed, _ = model.encode_Qformer_visual(img_array)
img_list.append(img_array_embed)
chat_state.append_message(chat_state.roles[0], "<Video><ImageHere></Video> ")

# Ask a question
user_message = "Describe this video in detail as a real estate agent."
chat.ask(user_message, chat_state)

# print("Chat State")
# print(chat_state)

# Get an answer
llm_answer, tokens, transition_scores, attn = chat.answer(
    conv = chat_state,
    img_list = img_list,
    num_beams = 1,
    temperature = 0.5,
    max_new_tokens = 1000,
    max_length = 2000,
)

# print(f"Answer: {llm_answer}")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(transition_scores.shape)
output_str = ""
for tok, score in zip(tokens, transition_scores[0]):
    # | token | token string | logits | probability
    score = score.cpu().numpy()
    prob = np.exp(score)
    token = model.llama_tokenizer.convert_ids_to_tokens([tok])
    token = token[0].replace("▁", " ")
    assert 0.0 <= prob <= 1.0
    if prob <= 0.2:
        output_str += bcolors.FAIL + token + bcolors.ENDC
    elif prob <= 0.5:
        output_str += bcolors.OKCYAN + token + bcolors.ENDC
    else:
        output_str += bcolors.OKGREEN + token + bcolors.ENDC

print(output_str)

## ATTENTION VIS


# attn_m = heterogenous_stack([
#     torch.tensor([
#         1 if i == j else 0
#         for j, token in enumerate(tokens)
#     ])
#     for i, token in enumerate(tokens)
# ] + list(map(aggregate_attention, attn)))

# for layer in attn:
#     print(layer)
#     break