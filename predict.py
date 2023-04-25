import torch
import clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
import json, os
from PIL import Image
from tqdm import tqdm
from matplotlib import font_manager
from application import get_clip_model, get_caption_model, generate_beam
from attention import interpret, get_clip_attention_model, show_heatmap_on_text, show_image_relevance

import warnings
warnings.simplefilter("ignore", UserWarning)

start_layer =  -1
start_layer_text =  -1
type_dict = {
    'caption_type': ['violation', 'status'],
    'violation_type': ['墜落', '機械', '物料', '感電', '防護具', '穿刺', '爆炸', '工作場所', '搬運']
}
font = font_manager.FontProperties(fname='/home/user/Documents/weilun/image_captioning/STHeiti-Medium.ttc')

def clip_classification(image, clip_model, preprocess, device):
    image_tensor = torch.tensor(preprocess(image)).to(device).unsqueeze(0)
    prediction = {}

    for key in ['caption_type', 'violation_type']:
        type = clip.tokenize(type_dict[key]).to(device)

        logits_per_image, _ = clip_model(image_tensor, type)
        prediction[key] = type_dict[key][torch.argmax(logits_per_image, dim=1).item()]

    return prediction['caption_type'], prediction['violation_type']

def export_plot(img, pred, gt, output_filename):    
    plt.title(f'pred: {"".join(pred)} \n ground truth: {gt}', fontproperties=font)
    plt.savefig(output_filename)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model_path = 'CLIP/models/clip_comb2_0_comb9_6_cap2_5.pt'
    clip_attention_model_path = 'CLIP/models/clip__comb9_999.pt'
    caption_model_path = 'CLIP_prefix_caption/models/coco_prefix_ct-0399.pt'
    
    clip_model, preprocess = get_clip_model(clip_model_path)
    clip_attention_model, preprocess = get_clip_attention_model(clip_attention_model_path, device)
    caption_model, tokenizer = get_caption_model(caption_model_path)

    output_path = f'output/attention'
    data = json.load(open('test.json'))
    prefix_length = 20
    attribute_length = 20
    for d in tqdm(data['annotations']):
        try:
            output_filename = os.path.join(output_path, d['file_name'].split('/')[-1])
            image = Image.open(d['file_name'])
            
            caption_type, violation_type = clip_classification(image, clip_model, preprocess, device)
            
            processed_image = preprocess(image).unsqueeze(0).to(device)
            prefix = clip_model.encode_image(processed_image).to(device, dtype=torch.float32)
            attribute = f'{caption_type} {violation_type} '
            
            encode_attribute = torch.tensor(tokenizer.encode(attribute), dtype=torch.int64)
            padding = attribute_length - encode_attribute.shape[0]
            encode_attribute = torch.cat((encode_attribute, torch.zeros(padding, dtype=torch.int64))).to(device)

            prefix_embed = caption_model.clip_project(prefix).reshape(1, prefix_length, -1)
            embedding_text = caption_model.gpt.transformer.wte(encode_attribute).unsqueeze(0)
            embedding_cat = torch.cat((prefix_embed, embedding_text), dim=1)

            caption = generate_beam(caption_model, tokenizer, embed=embedding_cat)

            tokenized_caption = clip.tokenize([caption]).to(device)
            R_text, R_image = interpret(model=clip_attention_model, image=processed_image, texts=tokenized_caption, device=device)

            show_heatmap_on_text(caption, tokenized_caption[0], R_text[0])
            show_image_relevance(R_image[0], processed_image, orig_image=image)

            gt_attribute = f"{d['caption_type']} {d['violation_type']} "
            plt.title(f"pred: {attribute + caption} \n ground truth: {gt_attribute + d['caption']}", fontproperties=font)
            plt.savefig(output_filename)
        except Exception as e:
            print(e)
            
if __name__ == '__main__':
    main()