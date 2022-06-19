from gc import collect
from re import A
import numpy as np
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

class ProjectionHead(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=512,
        dropout=0.2
    ):
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(projection_dim, projection_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        text = clip.tokenize(self.violation_list[item])[0]
        image = self.preprocess(Image.open(self.file_name[item]))
        print(text.shape, image.shape)
        exit()
        return image, text

    def __init__(self, preprocess, json_path: str, image_path: str):
        data = json.load(open(json_path, 'r'))

        self.preprocess = preprocess
        annotations = data["annotations"]
        self.ids = [caption["id"] for caption in annotations]
        self.caption_type = [annotation['caption_type'] for annotation in annotations]
        self.violation_type = [annotation['violation_type'] for annotation in annotations]
        self.violation_list = [annotation['violation_list'] for annotation in annotations]
        self.caption = [annotation['caption'] for annotation in annotations]
        self.file_name = [os.path.join(image_path, annotation['file_name']) for annotation in annotations]

        c = collections.Counter(self.caption_type)
        self.caption_type_dict = { k: [ a for a in annotations if a['caption_type'] == k] for k in c.keys() }

        print(self.caption_type_dict)
        # c = collections.Counter(self.violation_type)
        # print(c)

class ClipPairDataset(Dataset):

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        image = []
        text = []

        for k in self.pair_dict.keys():
            annotation = self.pair_dict[k][item % len(self.pair_dict[k])]
            
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
        print(c)
        # exit()
        self.dataset_len = c.most_common()[0][1]
        self.pair_dict = { k: [ a for a in annotations if a[key] == k] for k in list(c.keys())[:3] }

def main():
    print("Torch version:", torch.__version__)
    device = torch.device('cuda:0')

    model, preprocess = clip.load("ViT-B/32", device=device)

    model_path = 'models/clip_latest.pt'
    with open(model_path, 'rb') as opened_file: 
        model.load_state_dict(torch.load(opened_file, map_location="cpu"))

    batch_size = 1
    epochs = 10
    output_dir = 'models'
    output_prefix = 'clip'

    json_path = '../reju/reju.json'
    image_path = '../'
    lr = 1e-5
    num_warmup_steps = 5000
    save_every = 10

    # dataset = ClipCocoDataset(preprocess, json_path, image_path)
    dataset = ClipPairDataset(preprocess, json_path, image_path, 'caption_type')
    # dataset = data_utils.Subset(dataset, torch.arange(10))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    # reduce lr on plateau
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        loss_list = []
        for idx, (image, text) in enumerate(train_dataloader):
            model.zero_grad()
            image, text = image.to(device).squeeze(0), text.to(device).squeeze(0)

            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)

            # image_embed = np.dot(image_features, )
            # image_features = image_features / image_features.norm(dim=1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=1, keepdim=True)
            # print(image_features.shape, text_features.t().shape)
            # logits = torch.dot(image_features, text_features.t())
            # print(logits.shape)
            
            # print(image.shape, text.shape)
            logits_per_image, logits_per_text = model(image, text)
            label = torch.arange(logits_per_image.shape[0]).to(device)
            loss_i = criterion(logits_per_image, label)
            loss_t = criterion(logits_per_text, label)
            loss = (loss_i + loss_t) / 2
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tf_logger.add_scalar('loss', loss.item(), idx)
            progress.set_postfix({
                'loss': np.mean(loss_list),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
                })
            progress.update()

        progress.close()

        if epoch % save_every == 0:
            print('save')
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}_latest.pt"),
            )


if __name__ == '__main__':
    main()