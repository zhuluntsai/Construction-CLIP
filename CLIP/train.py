from gc import collect
from re import A
import numpy as np
from regex import P
import torch
from pkg_resources import packaging
import clip, os, sys, pickle, json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as nnf
from typing import Tuple, Optional, Union
import skimage.io as io
import clip
from PIL import Image
import collections
from torch.optim import lr_scheduler
import random
from itertools import combinations

import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

# https://tensorboard.dev/experiment/tNb8P9H2RnOzHNOAEYnayw/

model_name = 'clip'
tf_logger = SummaryWriter(log_dir=f'log/{model_name}')

seed = 567
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

class ClipPairDataset(Dataset):

    def __len__(self) -> int:
        return np.sum(self.cumulative_sizes)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        image = []
        text = []
        
        for i, length in enumerate(self.cumulative_sizes):
            if length <= item:
                item -= length
            else:
                break
        
        pair_dict = self.pair_list[i]
        for k in pair_dict.keys():
            annotation = pair_dict[k][item % len(pair_dict[k])]
            
            image_path = os.path.join(self.image_path, annotation['file_name'])
            image.append(self.preprocess(Image.open(image_path)))
            text.append(annotation[self.key])

        image =  torch.tensor(np.stack(image))
        text = clip.tokenize(text)
        return image, text

    def __init__(self, preprocess, json_path: str, image_path: str, key: str):
        data = json.load(open(json_path, 'r'))
        self.preprocess = preprocess
        self.image_path = image_path
        self.key = key

        annotations = data["annotations"]
        annotations = [annotation for annotation in annotations if annotation[key] != '']
        self.pair = [annotation[key] for annotation in annotations]

        c = collections.Counter(self.pair)
        combination = list(combinations(c.keys(), 2))
        self.combination = combination

        self.dataset_len = c.most_common()[0][1]
        pair_list = []
        for combine in combination:
            pair_dict = { k: [ a for a in annotations if a[key] == k] for k in combine }
            pair_list.append(pair_dict)

        self.pair_list = pair_list
        self.cumulative_sizes = [ max([len(p[d]) for d in p.keys()]) for p in self.pair_list ]

def main():
    print("Torch version:", torch.__version__)
    device = torch.device('cuda:0')

    model, preprocess = clip.load("ViT-B/32", device=device)

    # model_path = 'models/clip_latest.pt'
    # with open(model_path, 'rb') as opened_file: 
    #     model.load_state_dict(torch.load(opened_file, map_location="cpu"))

    batch_size = 1
    epochs = 10
    output_dir = 'models'
    output_prefix = 'clip'

    json_path = '../all.json'
    image_path = '../'
    lr = 1e-5
    num_warmup_steps = 5000
    save_every = 1

    dataset = ClipPairDataset(preprocess, json_path, image_path, 'violation_type')
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, no_deprecation_warning=True)
    # reduce lr on plateau
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        loss_list = []
        for idx, (image, text) in enumerate(train_dataloader):
            model.zero_grad()
            image, text = image.to(device).squeeze(0), text.to(device).squeeze(0)
            
            logits_per_image, logits_per_text = model(image, text)
            label = torch.arange(logits_per_image.shape[0]).to(device)

            loss_i = criterion(logits_per_image, label)
            loss_t = criterion(logits_per_text, label)
            loss = (loss_i + loss_t) / 2

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())
            tf_logger.add_scalar('loss', loss.item(), idx)
            progress.set_postfix({
                'loss': np.mean(loss_list),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
                })
            progress.update()

        progress.close()
        if (epoch + 1) % save_every == 0:
            print('save')
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}_latest.pt"),
            )


if __name__ == '__main__':
    main()