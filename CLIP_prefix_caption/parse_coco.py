import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import numpy as np

json_path = '../reju/reju.json'
image_path = '../'

def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./embedding/{clip_model_name}_reju_embedding.pkl"

    model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    # model_path = '../CLIP/models/clip_latest.pt'
    # with open(model_path, 'rb') as opened_file: 
    #     model.load_state_dict(torch.load(opened_file, map_location="cpu"))
    caption_types = {
        'status': '現況',
        'violation': '缺失'
    }
    violation_types = ['墜落', '防護具', '穿刺', '搬運', '感電', '爆炸', '工作場所', '物料', '機械']
    caption_type_token = clip.tokenize(list(caption_types.keys())).to(device)
    violation_type_token = clip.tokenize(violation_types).to(device)

    with open(json_path, 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data['annotations']))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data['annotations']))):
        annotations = data["annotations"][i]

        image = io.imread(os.path.join(image_path, annotations["file_name"]))
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = model.encode_image(image)
            
            logits_per_image, _ = model(image, caption_type_token)
            similarity = logits_per_image.softmax(dim=-1).cpu().numpy()
            index = np.argmax(similarity, axis=1)[0]
            caption_type = list(caption_types.values())[index]

            logits_per_image, _ = model(image, violation_type_token)
            similarity = logits_per_image.softmax(dim=-1).cpu().numpy()
            index = np.argmax(similarity, axis=1)[0]
            violation_type = violation_types[index]
            
        annotations["clip_embedding"] = i
        annotations['attribute'] = f'{caption_type} {violation_type} '
        
        all_embeddings.append(prefix)
        all_captions.append(annotations)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embeddings": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
