import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from transformers import CLIPProcessor, CLIPModel


json_path = '../fengyu/0_all.json'
image_path = '../fengyu/image'

def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./fengyu/{clip_model_name}_embedding.pkl"

    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(json_path, 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data['images']))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data['images']))):
        d = data['images'][i]
        a = data['annotations'][i]
        d['caption'] = a['caption']
        d['image_id'] = a['image_id']

        # image = io.imread(os.path.join(image_path, d["file_name"]))
        # image = preprocess(text=a['caption'] , images=Image.fromarray(image), return_tensors="pt", padding=True)#.unsqueeze(0).to(device)
        # image['pixel_values'] = image['pixel_values'][0].unsqueeze(0)
        # with torch.no_grad():a
        #     prefix_hf = clip_model(**image)

        image = io.imread(os.path.join(image_path, d["file_name"]))
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image)
        
        d["clip_embedding"] = i

        all_embeddings.append(prefix)
        all_captions.append(d)
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
