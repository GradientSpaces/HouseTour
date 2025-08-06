<p align="center">
  <h2 align="center"> HouseTour: A Virtual Real Estate A(I)gent </h2>
  <p align="center">
    <a>Ata Çelen</a><sup>1,2</sup>
    ·
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>1,3</sup>
    ·
    <a href="https://www.linkedin.com/in/d%C3%A1niel-bar%C3%A1th-3a489092/">Dániel Béla Baráth</a><sup>1,4</sup>
    ·
    <a href="https://ir0.github.io/">Iro Armeni</a><sup>2</sup>
  </p>
  <p align="center">
    <sup>1</sup>ETH Zürich    <sup>2</sup>Stanford University    <sup>3</sup>Microsoft Spatial AI Lab    <sup>4</sup>HUN-REN SZTAKI
  </p>
  <h3 align="center">

 [![arXiv](https://img.shields.io/badge/arXiv-blue?logo=arxiv&color=%23B31B1B)](https://placehold.co/600x400?text=Hello+World) 
 [![ProjectPage](https://img.shields.io/badge/Project_Page-HouseTour-blue)](https://house-tour.github.io/)
 [![Hugging Face (LCM) Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://placehold.co/600x400?text=Hello+World)
 [![HouseTour Dataset](https://img.shields.io/badge/HouseTour-Dataset-orange)](https://placehold.co/600x400?text=Hello+World)
 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
 <div align="center"></div>
</p>

## Abstract
We introduce **HouseTour**, a method for spatially-aware 3D camera trajectory and natural language summary generation from a collection of images depicting an existing 3D space. Unlike existing vision-language models (VLMs), which struggle with geometric reasoning, our approach generates smooth video trajectories via a diffusion process constrained by known camera poses and integrates this information into the VLM for 3D-grounded descriptions. We synthesize the final video using 3D Gaussian splatting to render novel views along the trajectory. To support this task, we present the HouseTour dataset, which includes over 1,200 house-tour videos with camera poses, 3D reconstructions, and real estate descriptions. Experiments demonstrate that incorporating 3D camera trajectories into the text generation process improves performance over methods handling each task independently. We evaluate both individual and end-to-end performance, introducing a new joint metric. Our work enables automated, professional-quality video creation for real estate and touristic applications without requiring specialized expertise or equipment.

## ✨ Key Features
- **Residual Diffuser** – diffusion-based planner that refines sparse poses into smooth, human-like camera paths. 
- **Qwen2-VL-3D** – VLM adapter that fuses vision + language + 3D tokens for real-estate-style scene summaries.
- **HouseTour Dataset** – >1 200 professionally filmed house-tour videos with 3D reconstructions & expert descriptions.

## Table of Contents
<ol>
  <li><a href="#⚙️-system-and-hardware">⚙️ System and Hardware</a></li>
  <li><a href="#🏠-housetour-dataset">🏠 HouseTour Dataset</a></li>
  <li><a href="#🖥️-setup">🖥️ Setup</a></li>
  <li><a href="#🏋️-training--inference">🏋️ Training & Inference</a></li>
  <li><a href="#🙏-acknowledgements">🙏 Acknowledgements</a></li>
  <li><a href="#📄-citation">📄 Citation</a></li>
</ol>


## ⚙️ System and Hardware
The code has been tested on:
```yaml
Ubuntu: 22.04 LTS
Python: 3.12.1
CUDA: 12.4
GPU: NVIDIA A100 SXM4 80GB
```

## 🏠 HouseTour Dataset
<p align="left">
  <img src="assets/recon_1.gif" alt="HouseTour Demo" width="400">
</p>

The HouseTour dataset is available at **_TODO_**

## 🖥️ Setup

### Environment Setup
```bash
$ git clone https://github.com/GradientSpaces/HouseTour.git
$ cd HouseTour

# Create an environment
$ conda create -n housetour python=3.12 -y
$ conda activate housetour
$ pip install -r requirements.txt

# Install the local package 'diffuser'
$ pip install -e .
```

### Downloading the Pretrained Models
You can download the pretrained models from the following links: **_TODO_**

## 🏋️ Training & Inference
### Residual Diffuser

To train the Residual Diffuser, you can use the following command:
```bash
python -m scripts.train_residual_diffuser
```
The configuration file for the Residual Diffuser is located at `config/residual_diffuser.py`. You can modify the training parameters in this file.

To run inference with the Residual Diffuser, use the following command:
```bash
python -m scripts.test_residual_diffuser
```

### Qwen2-VL-3D

1) To train the LoRA adapter on Qwen2-VL, you can use the following command:
```bash
python train_qwen2_vl.py \
    --data_dir "path/to/dataset" \
    --output-dir "training_checkpoints/lora_adapter"
```
2) To train the Qwen2-VL-3D model, you can use the following command:
```bash
python train_qwen2_vl_3d.py \
    --lora_adapter_dir "path/to/lora_adapter" \
    --data_dir "path/to/dataset" \
    --diffuser_dir "./checkpoints/residual-diffuser" \
    --traj_data "path/to/trajectories.jsonl"
```

To run inference with the Qwen2-VL-3D model on the evaluation set, use the following command:
```bash
python eval_qwen2_vl_3d.py \
    --model-path "./checkpoints/qwen2-vl-3d" \
    --traj-path "./checkpoints/residual-diffuser" \
    --traj-data "path/to/trajectories.jsonl"
```

## Contact
For any questions or issues, please open an issue on the GitHub repository or contact us via email: ata.celen@inf.ethz.ch

## 🙏 Acknowledgements
We thank the authors from [Diffuser](https://github.com/jannerm/diffuser) and [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) for open-sourcing their codebases.

## 📄 Citation
If you use this code or the dataset in your research, please cite our paper:
**_TODO_**
```bibtex
```
