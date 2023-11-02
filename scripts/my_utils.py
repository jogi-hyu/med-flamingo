import random
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .constants import *

def read_images_from_dir(img_dir: str, ext: str = "jpg") -> list:
    """
    Reads images from a directory
    """
    return [Image.open(img) for img in sorted(Path(img_dir).glob(f"*.{ext}"))]

def make_chunk(data, separator: str = " ") -> tuple:
    return (
        data['id'],
        f"{'<image>' * data['img_count']}{separator}"
        f"Medical History: {data['history']}{separator}"
        f"Image Findings: {data['findings']}{separator}"
        f"Final Diagnosis: {data['diagnosis']}.{separator}"
        f"<|endofchunk|>{separator*2}"
    )

def make_chunk_list(dataset: str, separator: str = " ") -> list:
    chunks = []

    if dataset == "kstr":
        datas = pd.read_csv(FILTERED_CSV[dataset], index_col=None)
        for i, data in datas.iterrows():
            chunks.append(make_chunk(data, separator))
    return chunks

def load_prompts(dataset:str, separator:str = "\n", shots:int = 0) -> list:
    """
    Loads the prompts
    """
    if shots < 0:
        raise ValueError("Number of shots must be a positive integer")
    if shots > MAX_SHOTS:
        raise ValueError(f"Number of shots must be less than or equal to {MAX_SHOTS}")

    text_chunks = make_chunk_list(dataset, separator)

    prompts = []
    for i, chunk in tqdm(text_chunks, desc="Loading prompts..."):
        split_chunk = chunk.split("Final Diagnosis: ")
        if shots == 0: # Zero-shot
            images = read_images_from_dir(IMAGE_PATH[dataset] / f"case_{i:04}")
            text = INSTRUCTION["zero"] + separator*2 + split_chunk[0]

        else: # Few-shot
            images = []
            text = INSTRUCTION["few"] + separator*2

            for j in range(shots):
                support_images = read_images_from_dir(FEW_SHOT_SUPPORT_SET[j]["img_dir"])
                support_text = make_chunk(FEW_SHOT_SUPPORT_SET[j], separator)[1]
                images += support_images
                text += support_text

            query_images = read_images_from_dir(IMAGE_PATH[dataset] / f"case_{i:04}")
            query_text = split_chunk[0]

            images += query_images
            text += query_text

        text += "Final Diagnosis: "
        prompts.append(
            {
                "id": i,
                "images": images,                        # list of PIL images
                "text": text,                            # string
                "gt": split_chunk[1].split(separator)[0] # string
            }
        )

    return prompts

def get_prompt_from_id(prompts: list, id: int) -> dict:
    """
    Gets the prompt with the given id
    """
    for prompt in prompts:
        if prompt["id"] == id:
            return prompt
    raise ValueError(f"Prompt with id {id} not found")