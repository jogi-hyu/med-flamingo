import os
import sys
sys.path.append('..')

import time
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from einops import repeat
from accelerate import Accelerator
from src.utils import FlamingoProcessor
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

from scripts.demo_utils import image_paths, clean_generation
from scripts.my_utils import load_prompts

def main():
    accelerator = Accelerator() #when using cpu: cpu=True

    device = accelerator.device
    
    print('Loading model...')

    # >>> add your local path to Llama-7B (v1) model here:
    llama_path = './models/llama-7b-hf'
    if not os.path.exists(llama_path):
        raise ValueError('Llama model not yet set up, please check README for instructions!')

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )
    # load med-flamingo checkpoint:
    checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt", cache_dir="../models/")
    print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    processor = FlamingoProcessor(tokenizer, image_processor)

    # go into eval model and prepare:
    model = accelerator.prepare(model)
    is_main_process = accelerator.is_main_process
    model.eval()

    """
    Step 1: Load images and text
    Step 2: Define multimodal few-shot prompt 
    """
    prompts = load_prompts(dataset="kstr", shots=0)
    # prompt = prompts[0]

    responses = []
    ids = []
    for prompt in tqdm(prompts, desc="Generating responses..."):
        if len(prompt['images']) == 0: continue
        ids.append(prompt['id'])
        """
        Step 3: Preprocess data 
        """
        # print('Preprocess data')

        pixels = processor.preprocess_images(prompt['images'])
        pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
        tokenized_data = processor.encode_text(prompt['text'])

        """
        Step 4: Generate response 
        """
        # actually run few-shot prompt through model:
        # print('Generate from multimodal few-shot prompt')
        generated_text = model.generate(
            vision_x=pixels.to(device),
            lang_x=tokenized_data["input_ids"].to(device),
            attention_mask=tokenized_data["attention_mask"].to(device),
            max_new_tokens=20,
        )
        response = processor.tokenizer.decode(generated_text[0])
        response = clean_generation(response)
        response = response.split("Final Diagnosis: ")[-1]
        responses.append(response)
    
    # save responses to csv:
    df = pd.DataFrame({'id':ids, 'response':responses})
    df.to_csv('responses.csv', index=False)

if __name__ == "__main__":
    main()