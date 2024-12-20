from autodq import eval_q_and_a_bool

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import argparse
import yaml
from tqdm import tqdm
import re

from grandtour.models.grandtour import GrandTour
from grandtour.conversation.chat import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2, StoppingCriteriaSub
from grandtour.datasets.data_utils import prepare_sample
from grandtour.datasets.house_tour_dataset import HouseTour_Dataset
from grandtour.processors.img_array_processors import ImgArrayProcessor

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
            
            images_embed, _ = model.encode_videoQformer_visual(images)
            
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
    model = GrandTour.from_config(cfg["model"]).to(device)

    ckpt_path = cfg["model"]["grandtour_model"]
    print("Loading model from ", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    dataset = HouseTour_Dataset(
        vis_processor=ImgArrayProcessor(image_size=224),
        text_processor=None,
        vis_root=cfg["data"]["vis_root"],
        ann_root=cfg["data"]["ann_root"],
        num_video_query_token=cfg["data"]["num_video_query_token"],
        tokenizer_name=cfg["data"]["tokenizer_name"],
        data_type=cfg["data"]["data_type"],
        model_type=cfg["data"]["model_type"],
        sample_type=cfg["data"]["sample_type"],
        max_txt_len=cfg["data"]["max_txt_len"],
        stride=cfg["data"]["stride"],
    )

    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=cfg["train"]["val_ratio"],
        shuffle=False,
    )

    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn = dataset.collater,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        persistent_workers=False,
        pin_memory=True,
        drop_last=False,
    )

    val_dict = evaluate_model(
        model,
        val_loader
    )

    print(val_dict)