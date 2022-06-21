from lib2to3.pgen2 import token
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    AdamW, 
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import numpy as np
import random

seed = 567
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.prefixes)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens = {}
        for k in self.captions.keys():
            tokens[k] = self.captions[k][item]
        tokens['prefix'] = self.prefixes[item].tolist()
        # print(tokens)
        return tokens

    def __init__(self, data_path: str,  prefix_length: int, tokenizer, gpt2_type: str = "gpt2"):
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = []
        self.captions = []
        self.caption2embedding = []

        for clip_embedding, caption in zip(all_data["clip_embedding"], all_data["captions"]):
            if caption['violation_list'] != '':
                self.captions.append(caption['violation_list'])
                self.caption2embedding.append(caption["clip_embedding"])
                self.prefixes.append(clip_embedding)

        # print(self.caption2embedding)
        # exit()
        self.captions = self.tokenizer(self.captions, max_length=32, padding='max_length', truncation=True)
        self.captions['labels'] = self.captions['input_ids']
        # self.captions['prefix'] = self.prefixes


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())

        print(layers)
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        encoder_outputs = self.t5.encoder(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        hidden_states = encoder_outputs[0]

        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        encoder_outputs.last_hidden_state = torch.cat((prefix_projections, hidden_states), dim=1)
        # print(tokens['input_ids'][0])
        # print(tokens['decoder_input_ids'][0])
        # print(tokens['labels'].shape)
        # print(tokens['labels'][0])
        # print(encoder_outputs.last_hidden_state.shape)
        
        device = torch.device('cuda:0')
        if tokens['labels'] is not None:
            dummy_token = self.get_dummy_token(tokens['labels'].shape[0], device)
            # print(dummy_token.shape)
            # print(dummy_token[0])
            labels = torch.cat((dummy_token, tokens['labels']), dim=1)
            # print(labels.shape)
            # print(labels[0])
            
        
        mask = tokens['input_ids'].ge(0)
        mask = mask.float().to(device)
        mask = torch.cat((torch.ones(mask.shape[0], self.prefix_length).to(device), mask), dim=1)

        # print(decoder_input_ids[0])
        output = self.t5(
            # inputs_embeds=encoder_outputs,
            encoder_outputs=encoder_outputs,
            # attention_mask=mask,
            decoder_input_ids=labels,
        )
        return output

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        config = AutoConfig.from_pretrained('google/mt5-small')
        self.t5 = MT5ForConditionalGeneration.from_pretrained('google/mt5-small', config=config)
        self.gpt_embedding_size = self.t5.model_dim
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        # self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, tokenizer, args,
          lr: float = 1e-4, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    save_every = 10

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    label_pad_token_id = -100 
    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model.t5,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )

    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        loss_list = []
        for idx, tokens in enumerate(train_dataloader):
            model.zero_grad()
            tokens = tokens.to(device)
            prefix = tokens.pop('prefix')
            outputs = model(tokens, prefix)

            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            # print(outputs.logits.shape)
            # print(logits.shape)
            # print(logits.reshape(-1, logits.shape[-1]).shape)
            # print(tokens['input_ids'].shape)
            # print(tokens['input_ids'].flatten().shape)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tokens['input_ids'].flatten())
            # print(loss)
            # exit()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
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
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./embedding/ViT-B_32_reju_embedding.pkl')
    parser.add_argument('--out_dir', default='./models')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length

    tokenizer = AutoTokenizer.from_pretrained('google/mt5-small', use_fast=True)
    dataset = ClipCocoDataset(args.data, prefix_length, tokenizer)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    train(dataset, model, tokenizer, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()
