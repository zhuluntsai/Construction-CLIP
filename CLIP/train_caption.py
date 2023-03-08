from gc import collect
from re import A
import numpy as np
from regex import P
from tomlkit import item
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


seed = 567
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

class ClipPairDataset(Dataset):

    def __len__(self) -> int:
        return len(self.pair_list)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        annotation = self.pair_list[item]    
        image_path = os.path.join(self.image_path, annotation['file_name'])

        image =  torch.tensor(self.preprocess(Image.open(image_path)))
        text = clip.tokenize(annotation[self.key])
        return image, text

    def __init__(self, preprocess, json_path: str, image_path: str, train_ratio: str, key: str, split: str):
        self.preprocess = preprocess
        self.image_path = image_path
        self.key = key

        data = json.load(open(json_path, 'r'))
        annotations = data["annotations"]
        annotations = [annotation for annotation in annotations if annotation[key] != '']

        train_amount = int(len(annotations) * train_ratio)
        pair_list = {'train': annotations[:train_amount], 'test': annotations[:train_amount]}

        self.pair_list = pair_list[split]
        print(self.pair_list[0])
        print(len(self.pair_list))

        
def main():
    print("Torch version:", torch.__version__)
    device = torch.device('cuda:0')

    model, preprocess = clip.load("ViT-B/32", device=device)

    model_prefix = ''
    # model_prefix = 'balance_comb2_699'
    model_path = f'models/clip_{model_prefix}.pt'
    # with open(model_path, 'rb') as opened_file: 
    #     model.load_state_dict(torch.load(opened_file, map_location="cpu"))
    # model_path = f'models/clip_comb2_0_comb9_6.pt'

    # step = 3490
    # step = 41670
    step = 0
    test_step = 0
    combination_num = 9
    key = 'violation_list' # caption_type violation_type
    batch_size = 1
    epochs = 1000
    output_dir = 'models'
    output_prefix = 'clip'

    json_path = '../all.json'
    image_path = '../'
    lr = 1e-5
    num_warmup_steps = 5000
    save_every = 100
    train_ratio = 0.8

    model_name = f'{model_prefix}_comb{combination_num}'
    tf_logger = SummaryWriter(log_dir=f'log/{model_name}')

    train_dataset = ClipPairDataset(preprocess, json_path, image_path, train_ratio, key, 'train')
    test_dataset = ClipPairDataset(preprocess, json_path, image_path, train_ratio, key, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
        accuracy_list = []
        model.train()
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

            accuracy = sum(torch.argmax(logits_per_image, dim=1) == label) / len(label)

            loss_list.append(loss.item())
            accuracy_list.append(accuracy.item())
            tf_logger.add_scalar('training loss', loss.item(), step)
            tf_logger.add_scalar('training accuracy', accuracy.item(), step)
            tf_logger.add_scalar('learning rate', scheduler.optimizer.param_groups[0]['lr'], step)
            step += 1
            progress.set_postfix({
                'loss': np.mean(loss_list),
                'acc': np.mean(accuracy_list),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
                })
            progress.update()

        progress = tqdm(total=len(test_dataloader), desc=output_prefix)
        loss_list = []
        accuracy_list = []
        model.eval()
        with torch.no_grad():
            for idx, (image, text) in enumerate(test_dataloader):
                image, text = image.to(device).squeeze(0), text.to(device).squeeze(0)
                
                logits_per_image, logits_per_text = model(image, text)
                label = torch.arange(logits_per_image.shape[0]).to(device)

                accuracy = sum(torch.argmax(logits_per_image, dim=1) == label) / len(label)

                tf_logger.add_scalar('testing accuracy', accuracy.item(), test_step)
                test_step += 1
                accuracy_list.append(accuracy.item())
                progress.set_postfix({
                    'acc': np.mean(accuracy_list),
                    })
                progress.update()
        
        progress.close()

        if (epoch + 1) % save_every == 0:
            print('save')
            torch.save(
                model.state_dict(),
                # os.path.join(output_dir, f"{output_prefix}_{model_prefix}_cap{combination_num}_{epoch}.pt"),
                os.path.join(output_dir, f"{output_prefix}_{model_prefix}_comb{combination_num}_{epoch}.pt"),
            )


if __name__ == '__main__':
    main()