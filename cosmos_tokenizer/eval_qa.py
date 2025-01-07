from autodq import eval_q_and_a_bool

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import argparse
import yaml
from tqdm import tqdm
import re
import os

from cosmos_tokenizer.models.model import CosmosLlama
from cosmos_tokenizer.conversation.chat import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2, StoppingCriteriaSub
from cosmos_tokenizer.datasets.data_utils import prepare_sample
from cosmos_tokenizer.datasets.house_tour_dataset import HouseTour_Dataset
from cosmos_tokenizer.processors.img_array_processors import ImgArrayProcessor

def find_token_sequence(input_ids, token_sequence):
    """
    Find the start index of a token sequence in input_ids.

    Args:
    - input_ids (torch.Tensor): A 1D tensor of token IDs representing the input sequence.
    - token_sequence (list of int): The list of token IDs to search for.

    Returns:
    - index (int): The starting index of the token sequence, or -1 if not found.
    """
    input_len = input_ids.size(0)
    seq_len = len(token_sequence)

    for i in range(input_len - seq_len + 1):
        if (input_ids[i:i + seq_len] == torch.tensor(token_sequence).to(input_ids.device)).all():
            return i + seq_len
    return -1


def evaluate_model(model, data_loader, cuda_enabled=False):
    """
    Evaluate the model and calculate accuracy and F1 score.
    """
    model.eval()  # Set model to evaluation mode
    correct_preds = 0
    incorrect_preds = 0 

    print("Evaluating...")
    with torch.no_grad():
        for samples in tqdm(data_loader):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            images = samples['images'].to(model.device)
            input_ids = samples['input_ids'].to(model.device)
            
            # print(f"Images Shape: {images.shape}")
            images_embed, _ = model.encode_cosmos_visual(images)
            
            img_tok_indices = (input_ids[0] == model.IMAGE_PATCH_TOKEN_ID).nonzero(as_tuple=True)[0]
            seg_prev, seg_after = input_ids[:, : img_tok_indices[0]], input_ids[:, img_tok_indices[-1] + 1: ]

            # Seg_after should stop after [/INST]
            end_inst_idx = find_token_sequence(seg_after[0], [518, 29914, 25580, 29962]) #[/INST]
            seg_after = seg_after[:, :end_inst_idx]

            
            question = re.search(r"</Video>(.*?)\[/INST\]", model.llama_tokenizer.decode(seg_after[0]))
            question = question.group(1).strip() if question else None
            print(f"Q: {question}")

            if model.lora:
                seg_embs = [model.llama_model.get_base_model().model.embed_tokens(seg_t) for seg_t in [seg_prev, seg_after]]
            else:
                seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in [seg_prev, seg_after]]
            
            embs = torch.cat((seg_embs[0], images_embed, seg_embs[1]), dim=1)

            max_length=2000
            max_new_tokens=100

            current_max_len = embs.shape[1] + max_new_tokens
            if current_max_len - max_length > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                      'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - max_length)


            embs = embs[:, begin_idx:]
            stop_words_ids = [torch.tensor([2]).to(model.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

            output = model.llama_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )

            print(f"Pred: {model.llama_tokenizer.decode(output[0])}")
            labels_idx = (samples['labels'] != -100).nonzero(as_tuple=True)
            labels = samples['labels'][labels_idx]
            print(f"Label: {model.llama_tokenizer.decode(labels)}")

            autodq_eval = eval_q_and_a_bool(
                question = question, 
                pred = model.llama_tokenizer.decode(output[0]), 
                gt = model.llama_tokenizer.decode(labels)
            )

            if autodq_eval is None:
                continue

            print(f"AutoDQ: {autodq_eval}")

            if autodq_eval['correct']:
                correct_preds += 1
            else:
                incorrect_preds += 1

            print(f"Rolling Correctness : {correct_preds / (correct_preds + incorrect_preds)}")

    return {"accuracy": correct_preds / (correct_preds + incorrect_preds)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True)
    args = parser.parse_args()
    cfg_path = args.cfg_path    

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CosmosLlama(cfg).to(device)

    ckpt_path = "results/train_cosmos_qa_e1_iter15000_loss0.756.pth"
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    dataset = HouseTour_Dataset(
        vis_processor=ImgArrayProcessor(image_size=-1),
        text_processor=None,
        vis_root=cfg["data"]["vis_root"],
        ann_root=cfg["data"]["ann_root"],
        num_video_query_token=cfg["data"]["num_video_query_token"],
        tokenizer_name=cfg["data"]["tokenizer_name"],
        data_type=cfg["data"]["data_type"],
        model_type=cfg["data"]["model_type"],
        sample_type=cfg["data"]["sample_type"],
        max_txt_len=cfg["data"]["max_txt_len"],
        resize_size=(512, 288)
        # stride=cfg["model"]["context_size"],
    )

    #Read the validation scene ids
    with open(os.path.join(cfg["data"]["vis_root"], "val_indexes.txt"), "r") as f:
        val_scenes = f.read()
        val_scenes = val_scenes.split("\n")
        val_scenes = [int(i) for i in val_scenes]

    val_idxs = [i for i in range(len(dataset)) if dataset.get_scene_id(i) in val_scenes]

    val_dataset = Subset(dataset, val_idxs)
    
    val_loader = DataLoader(
        val_dataset,
        collate_fn = dataset.collater,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        persistent_workers=False,
        pin_memory=True,
        drop_last=False,
    )

    val_dict = evaluate_model(
        model,
        val_loader,
        cuda_enabled=True
    )

    print(val_dict)