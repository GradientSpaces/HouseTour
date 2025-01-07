import yaml
import os
from PIL import Image
import torch
from torchvision import transforms
import einops
from torch.amp import autocast as autocast
import json

from cosmos_tokenizer.models.model import CosmosLlama
from cosmos_tokenizer.conversation.chat import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2
from cosmos_tokenizer.processors.img_array_processors import ImgArrayProcessor

anno_path = "/scratch/users/atacelen/Reconstructions3D/annotations.json"
with open(anno_path, "r") as f:
    anno = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = yaml.safe_load(open("cosmos_tokenizer/configs/cosmos_llama_eval.yaml"))
# cfg = yaml.safe_load(open("cosmos_tokenizer/configs/timechat.yaml"))
model = CosmosLlama(cfg).to(device)

ckpt_path = "results/cosmos_llama_init_e10_r8.pth"
model.load_state_dict(torch.load(ckpt_path))
model.eval()

vis_processor = ImgArrayProcessor.from_config(cfg["preprocess"]["vis_processor"]["eval"])

chat = Chat(model, vis_processor, device)
print("Chat initialized")

# Upload the image
chat_state = conv_llava_llama_2.copy()
chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

video_idx = 51
# anno_idx = next((item for item in anno if item.get("scene_id") == video_idx), None)
# candidates = anno_idx["candidates"]
# Optionally: Take the candidates from the candidates.txt
with open(f"/scratch/users/atacelen/Reconstructions3D/{video_idx}_video/candidates.txt", "r") as f:
    candidates = f.read().split("\n")


# Prepare the image array from candidates
img_list = []
keyframe_dir = f"/scratch/users/atacelen/Reconstructions3D/{video_idx}_video/keyframes_resized"

img_array_path = [os.path.join(keyframe_dir, c) for c in candidates]
img_array = vis_processor(img_array_path)

# Add the array to the chat
img_array = img_array.unsqueeze(0).to(device)
img_array_embed, _ = model.encode_cosmos_visual(img_array)
img_list.append(img_array_embed)
chat_state.append_message(chat_state.roles[0], "<Video><ImageHere></Video> ")

# Ask a question
user_message = "Describe this video in detail as a real estate agent."
chat.ask(user_message, chat_state)

# Get an answer
llm_answer = chat.answer(
    conv = chat_state,
    img_list = img_list,
    num_beams = 1,
    temperature = 0.1,
    max_new_tokens = 1000,
    max_length = 2000,
)[0]

print(f"Answer: {llm_answer}")