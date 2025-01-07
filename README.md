# 📚 Master Thesis Code Repository: Touring Real Estates using Sparse Image Observations​

<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
  <img src="./assets/recon_1.gif" style="width: 50%;">
</div>


## 📝 Abstract
> This repository contains the implementation, data, and tools developed for my master's thesis, **Touring Real Estates using Sparse Image Observations​**.  


## 📂 Repository Structure

### Cosmos Tokenizer
The folder *cosmos_tokenizer* holds the code repository for the Cosmos Tokennizer + LLaMa model
```
├── cosmos_tokenizer/  # Cosmos Tokennizer + LLaMa model files
├── autodq.py          # Script for the AutoDQ LLaMa-based evaluation 
├── eval_capt.py       # Script for the Dense Video Captioning evaluation
├── eval_qa.py         # Script for the Open VQA evaluation
├── eval.py            # Script for manual evaluation
├── train_tf.py        # Script for T/F training
├── train_q_and_a.py   # Script for Open VQA training
└── train.py           # Script for Dense Video Captioning training 
```

### Data Acquisition
The folder *data_acquisition* holds the code repository for the Data Acquisition Pipeline
```
├── singularity/                    # Cosmos Tokennizer + LLaMa model files
    ├── colmap_n_glomap.def         # Singularity recipe for COLMAP and GLOMAP installation on the cluster
    └── pipe.def                    # Singularity recipe for the Data Acquisition environment on the cluster
├── anyloc_retrieval.py             # 'AnyLoc' based Image Retrieval (experimental)
├── blip_vqa.py                     # Removal of outdoor frames with BLIP2 Visual Question Answering
├── build_vocab_tree.py             # Building a custom Vocabulary Tree for Image Retrieval (experimental)
├── database_matches_to_txt.py      # Helper script to write the retrieved image pairs to a .txt file
├── match_keyframes_mast3r_vocab.py # Finding two-view correspondences with Mast3r
├── pipe_cluster.sh                 # Bash script for Data Acquisition 
├── resize_keyframes.py             # Resizing the keyframes for further processing
├── run.py                          # Generating a Dense Point Cloud after COLMAP mapping with Dust3r
└── write_keypoints_h5.py           # Helper script to write the keypoints to an H5 file
```

### Datasets
The folder *datasets* holds the textual data used for the T/F, Open VQA and Dense Multi-Image Captioning tasks
```
├── annotations_cleaned_v2.json     # The dataset used for Dense Multi-Image Captioning
├── annotations_q_and_a.json        # The dataset used for Open VQA
├── annotations_true_false.json     # The dataset used for T/F
├── multi_image_pretrain_w8.json    # The dataset used for the Multi-Image pretraining (8 images) 
└── multi_image_pretrain_w8.json    # The dataset used for the Multi-Image pretraining (16 images)
```

### QFormer + Sliding Window Video-QFormer & Multi-Image QFormer + LLaMa
The folder *housetour* holds the code repository for the QFormer + Sliding Window Video-QFormer & Multi-Image QFormer + LLaMa models
```
├── grandtour/                  # QFormer + Sliding Window Video-QFormer & Multi-Image QFormer + LLaMa model files
├── train_scripts/              
    ├── pretrain.py             # First Stage Pretraining Script for the Multi-Image QFormer
    ├── train_blip2_llama.py    # Second Stage LLM training Script for the Multi-Image QFormer
    ├── train_captioning.py     # Second Stage LLM training Script for the Multi-Image QFormer
    ├── train_dist_multi_n.py   # Multi-node parallel model training script
    ├── train_dist_one_n.py     # Single-node parallel model training script
    ├── train_full.py           # Training simulatenously on the T/F + Open VQA + Captioning datasets
    ├── train_q_and_a.py        # Training Script for Open VQA with QF+SWV-QF
    ├── train_true_or_false.py  # Training Script for T/F with QF+SWV-QF
    └── train.py                # Training Script for Dense Multi-Image Captioning with QF+SWV-QF
└── eval_scripts/               # Evaluation Scripts for the T/F, Open VQA and Dense Multi-Image Captioning tasks
```
