import torch
import torch.nn as nn
from transformers import LlamaTokenizer
import einops
import copy

from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.models.modeling_llama import LlamaForCausalLM


class CosmosLlama(nn.Module):
    def __init__(self, cfg):
        super(CosmosLlama, self).__init__()
        
        self.cfg = cfg
        self.cosmos_encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{cfg["model"]["cosmos_encoder"]}/encoder.jit')

        print("Loading Llama Tokenizer...")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(cfg["model"]["llama_model"], use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        print("Loading the Llama Model...")
        self.llama_model = LlamaForCausalLM.from_pretrained(
            cfg["model"]["llama_model"],
            torch_dtype=torch.bfloat16,
        ) 

        if cfg["model"]["use_grad_checkpoint"]:
            print("use gradient checkpointing for LLAMA")
            self.llama_model.gradient_checkpointing_enable()

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.lora = cfg["model"]["lora"]
        if self.lora:
            print(f'Using LORA (lora_alpha={cfg["model"]["lora_alpha"]})')
            from peft import LoraConfig, get_peft_model, TaskType
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=cfg["model"]["lora_inference_mode"],
                r=cfg["model"]["lora_r"],
                lora_alpha=cfg["model"]["lora_alpha"],
                lora_dropout=0.1,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()
        
        print("Loading Llama Proj.")
        #TODO
        self.llama_proj = nn.Linear(
            4 * 36 * 64, self.llama_model.config.hidden_size
        )

        if cfg["model"]["frozen_llama_proj"]:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            print('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            print('LLAMA proj is not frozen')

        print('Loading llama_proj Done')
        self.max_txt_len = cfg["model"]["max_txt_len"]
        self.end_sym = cfg["model"]["end_sym"]
        self.device = list(self.parameters())[0].device

        self.conv1x1 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1)
        self.video_frame_position_embedding = nn.Embedding(cfg["model"]["max_frame_pos"], 36 * 64)
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def encode_cosmos_visual(self, image):
        device = image.device
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')

        with self.maybe_autocast():
            (image_embeds,) = self.cosmos_encoder.encode(image)
        
        position_ids = torch.arange(time_length, dtype=torch.long, device=image_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        frame_position_embeddings = self.video_frame_position_embedding(position_ids)
        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)

        with self.maybe_autocast():
            image_embeds_conv = self.conv1x1(image_embeds)
        # print(f"image_embeds_conv: {image_embeds_conv.shape}")        
        image_embeds_conv = einops.rearrange(image_embeds_conv, '(b t) n q h -> b t n (q h)',b=batch_size,t=time_length)
        # print(f"image_embeds_conv: {image_embeds_conv.shape}")
        # print(f"frame_position_embeddings: {frame_position_embeddings.shape}")
        frames_hidden_state = image_embeds_conv + frame_position_embeddings
        frames_hidden_state = einops.rearrange(frames_hidden_state, 'b t n qh -> b t (n qh)')
        frames_hidden_state = self.llama_proj(frames_hidden_state)

        return frames_hidden_state, torch.ones(batch_size, time_length).to(device)

    def forward(self, samples):
        if 'conv_type' in samples.keys() and samples['conv_type'] == 'multi':
            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            # Convert the input ids back to tokens
            # print(f"Input IDs: {self.llama_tokenizer.convert_ids_to_tokens(input_ids[0])}")
            if isinstance(image, list):  # nb of frames of some videos is less than ${num_frm}
                img_embeds_list, atts_img_list, num_patch_tokens_list = [], [], []
                for img in image:
                    img = img.unsqueeze(0)
                    if len(img.size()) == 4:
                        time = 1
                        img = einops.repeat(img, 'b c h w -> b c t h w', t=time)

                    num_patch_tokens = 96
                    img_embeds, atts_img = self.encode_cosmos_visual(img)
                    img_embeds_list.append(img_embeds)
                    atts_img_list.append(atts_img)
                    num_patch_tokens_list.append(num_patch_tokens)
                img_embeds = img_embeds_list
                atts_img = atts_img_list
            else:  # nb of frames of all videos is ${num_frm}
                if len(image.size()) == 4:
                    time = 1
                    image = einops.repeat(image, 'b c h w -> b c t h w', t=time)

                num_patch_tokens = 96
                img_embeds, atts_img = self.encode_cosmos_visual(image)

            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            if self.lora:
                temp_input_embedding = self.llama_model.get_base_model().model.embed_tokens(temp_input_ids)
            else:
                temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]  # [num_video_query_token, dim]
                if isinstance(image, list):
                    cur_image_features = cur_image_features.squeeze(0)
                    num_patch_tokens = num_patch_tokens_list[cur_image_idx]
                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                    raise ValueError(
                        "The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patch_tokens,
                                                   device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")

                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features,
                                                  cur_input_embeds[mask_index_start + num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)

                cur_image_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            print(f"Targets Shape: {targets.size()}")
            print(f"Att. Mask Shape: {attention_mask.size()}")
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            return {"loss": loss}
        else:
            image = samples["image"]
            _device = image.device
            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w', t=time)
            img_embeds, atts_img = self.encode_cosmos_visual(image)
            print("Image Encoding Done!")

            self.llama_tokenizer.padding_side = "right"

            text = [t + self.end_sym for t in samples["text_input"]]

            print("Text: ", text)

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                           dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                             dtype=to_regress_tokens.input_ids.dtype,
                             device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            if self.lora:
                bos_embeds = self.llama_model.get_base_model().model.embed_tokens(bos)
                to_regress_embeds = self.llama_model.get_base_model().model.embed_tokens(to_regress_tokens.input_ids)
            else:
                bos_embeds = self.llama_model.model.embed_tokens(bos)
                to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            atts_bos = atts_img[:, :1]

            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)


            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
        return {"loss": loss} 
