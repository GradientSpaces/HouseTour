import os

model = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir="/scratch/users/atacelen/.cache/"
os.environ['HF_HOME'] = cache_dir

from tqdm import tqdm
import torch
import yaml

from eval_tf import evaluate_model as evaluate_model_tf
from eval_qa import evaluate_model as evaluate_model_qa
from grandtour.datasets.data_utils import prepare_sample
from grandtour.models.grandtour import GrandTour
from grandtour.datasets.house_tour_dataset import HouseTour_Dataset
from grandtour.processors.img_array_processors import ImgArrayProcessor


def evaluate_model_capt(model, data_loader, cuda_enabled=False):
    val_loss_capt = 0.0
    with torch.no_grad():
        for samples in tqdm(data_loader, desc="Evaluating Captioning"):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            with torch.cuda.amp.autocast(enabled=cuda_enabled):
                loss = model(samples)["loss"]
            
            val_loss_capt += loss.item()
    
    return {"loss" : val_loss_capt / len(data_loader)}



if  __name__ == "__main__":
    cfg_path = "grandtour/configs/grandtour_eval_full.yaml"    

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrandTour.from_config(cfg["model"]).to(device)

    ckpt_path = "results/train_full_step10000.pth"
    print(f"Loading from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # Load the dataset
    full_dataset_tf = HouseTour_Dataset(
        vis_processor=ImgArrayProcessor(image_size=model.visual_encoder.image_size),
        text_processor=None,
        vis_root=cfg["data"]["vis_root"],
        ann_root=cfg["data"]["ann_root_tf"],
        num_video_query_token=cfg["data"]["num_video_query_token"],
        tokenizer_name=cfg["data"]["tokenizer_name"],
        data_type=cfg["data"]["data_type"],
        model_type=cfg["data"]["model_type"],
        sample_type=cfg["data"]["sample_type"],
        max_txt_len=cfg["data"]["max_txt_len"],
        stride=cfg["data"]["stride"],
    )
    full_dataset_qa = HouseTour_Dataset(
        vis_processor=ImgArrayProcessor(image_size=model.visual_encoder.image_size),
        text_processor=None,
        vis_root=cfg["data"]["vis_root"],
        ann_root=cfg["data"]["ann_root_qa"],
        num_video_query_token=cfg["data"]["num_video_query_token"],
        tokenizer_name=cfg["data"]["tokenizer_name"],
        data_type=cfg["data"]["data_type"],
        model_type=cfg["data"]["model_type"],
        sample_type=cfg["data"]["sample_type"],
        max_txt_len=cfg["data"]["max_txt_len"],
        stride=cfg["data"]["stride"],
    )
    full_dataset_capt = HouseTour_Dataset(
        vis_processor=ImgArrayProcessor(image_size=model.visual_encoder.image_size),
        text_processor=None,
        vis_root=cfg["data"]["vis_root"],
        ann_root=cfg["data"]["ann_root_capt"],
        num_video_query_token=cfg["data"]["num_video_query_token"],
        tokenizer_name=cfg["data"]["tokenizer_name"],
        data_type=cfg["data"]["data_type"],
        model_type=cfg["data"]["model_type"],
        sample_type=cfg["data"]["sample_type"],
        max_txt_len=cfg["data"]["max_txt_len"],
        stride=cfg["data"]["stride"],
    )
    train_dataset_tf, val_dataset_tf = torch.utils.data.random_split(full_dataset_tf, [0.98, 0.02])
    train_dataset_qa, val_dataset_qa = torch.utils.data.random_split(full_dataset_qa, [0.98, 0.02])
    train_dataset_capt, val_dataset_capt = torch.utils.data.random_split(full_dataset_capt, [0.9, 0.1])

    val_loader_tf = torch.utils.data.DataLoader(
        val_dataset_tf,
        collate_fn = full_dataset_tf.collater,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    val_loader_qa = torch.utils.data.DataLoader(
        val_dataset_qa,
        collate_fn = full_dataset_qa.collater,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    val_loader_capt = torch.utils.data.DataLoader(
        val_dataset_capt,
        collate_fn = full_dataset_capt.collater,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        drop_last=False,
    )

    # tf_dict = evaluate_model_tf(model, val_loader_tf, cuda_enabled=True)
    # print(f"T/F Eval: {tf_dict}")
    # qa_dict = evaluate_model_qa(model, val_loader_qa, cuda_enabled=True)
    # print(f"Q&A Eval: {qa_dict}")
    capt_dict = evaluate_model_capt(model, val_loader_capt, cuda_enabled=True)
    print(f"Captioning Loss: {capt_dict}")
