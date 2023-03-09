import numpy as np
import torch
import torchvision
import json
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from flask import Flask, jsonify, request
import clip
from transformers import GPT2LMHeadModel, AutoTokenizer
from typing import Tuple, Optional, Union
import torch.nn as nn

def get_object_model(model_path):
    num_classes = 7
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location='cpu')
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.eval()
    return model

def get_clip_model(model_path):
    model, preprocess = clip.load("ViT-B/32", device=device)
    with open(model_path, 'rb') as opened_file: 
        model.load_state_dict(torch.load(opened_file, map_location="cpu"))
    
    return model, preprocess

def get_caption_model(model_path):
    prefix_length = 20
    prefix_length_clip = 20
    prefix_dim = 512
    gpt2_type = 'ckiplab/gpt2-base-chinese'

    model = ClipCaptionModel(prefix_length, clip_length=prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=8, gpt2_type=gpt2_type)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(gpt2_type)

    return model, tokenizer

application = Flask(__name__)

def object_detection(image):
    # model = get_model()
    
    ## transform image to tensor
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    # predict
    with torch.no_grad():
        prediction = object_model(image_tensor.to(device))[0]

    output = {}
    output['boxes'] = prediction['boxes'].cpu().numpy().tolist()
    output['scores'] = prediction['scores'].cpu().numpy().tolist()
    output['labels'] = []
    prediction_label_ids = prediction['labels'].cpu().numpy().tolist()
    for id in prediction_label_ids:
        output['labels'].append(get_label(str(id)))

    return output

def get_label(id):
    f = open(label_path)
    data = json.load(f)
    labels = data["labels"]
    # Closing file
    f.close()
    return labels[id]

def clip_classification(image):
    image_tensor = torch.tensor(preprocess(image)).to(device).unsqueeze(0)
    prediction = {}

    for key in ['caption_type', 'violation_type']:
        type = clip.tokenize(type_dict[key]).to(device)

        logits_per_image, _ = clip_model(image_tensor, type)
        prediction[key] = type_dict[key][torch.argmax(logits_per_image, dim=1).item()]

    return prediction['caption_type'], prediction['violation_type']

def image_caption(image, caption_type, violation_type):
    prefix_length = 20
    attribute_length = 20
    
    processed_image = preprocess(image).unsqueeze(0).to(device)
    prefix = clip_model.encode_image(processed_image).to(device, dtype=torch.float32)
    attribute = f'{caption_type} {violation_type} '
    
    encode_attribute = torch.tensor(tokenizer.encode(attribute), dtype=torch.int64)
    padding = attribute_length - encode_attribute.shape[0]
    encode_attribute = torch.cat((encode_attribute, torch.zeros(padding, dtype=torch.int64))).to(device)

    prefix_embed = caption_model.clip_project(prefix).reshape(1, prefix_length, -1)
    embedding_text = caption_model.gpt.transformer.wte(encode_attribute).unsqueeze(0)
    embedding_cat = torch.cat((prefix_embed, embedding_text), dim=1)

    return generate_beam(caption_model, tokenizer, embed=embedding_cat)

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
        self.model = nn.Sequential(*layers)

class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, attribute: torch.Tensor, mask: Optional[torch.Tensor]=None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = torch.cat((attribute, tokens), dim=1)
        embedding_text = self.gpt.transformer.wte(embedding_text)

        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, gpt2_type: str = ''):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_type)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))

def generate_beam(
    model,
    tokenizer,
    beam_size: int = 3,
    prompt=None,
    embed=None,
    entry_length=100,
    temperature=0.5,
    stop_token: int = 102,
):

    model.eval()
    stop_token_index = stop_token
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated, output_attentions=True)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_text = [output_texts[i] for i in order][0]

    output_text.replace('CLS', '').replace('SEP', '').replace(' ', '')

    return output_text

@application.route('/predict', methods=['POST'])
def predict():
    i = 0
    if request.method == 'POST':
        file = request.files['file']
        image_extensions=['ras', 'xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm', 'xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm']
        if file.filename.split('.')[1] not in image_extensions:
            return jsonify('Please upload an appropriate image file')

        saveLocation = file.filename
        file.save(saveLocation)
        image = Image.open(saveLocation)

        prediction = object_detection(image)
        caption_type, violation_type = clip_classification(image)
        caption = image_caption(image, caption_type, violation_type) 
        
        return jsonify({"boxes": prediction['boxes'], 
            "labels": prediction['labels'], 
            "scores": prediction['scores'], 
            "caption_type": caption_type,
            "violation_type": violation_type,
            'caption': caption,
            })

@application.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. """
    return jsonify({"response": __name__})

@application.route("/")
def home():
    return "Hello, World!"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

object_model_path = '../pytorch/model_final.pth'
clip_model_path = '../image_captioning/CLIP/models/clip_comb2_0_comb9_6_cap2_5.pt'
caption_model_path = '../image_captioning/CLIP_prefix_caption/models/coco_prefix_ct-0399.pt'
label_path = '../pytorch/labels.json'

object_model = get_object_model(object_model_path)
clip_model, preprocess = get_clip_model(clip_model_path)
caption_model, tokenizer = get_caption_model(caption_model_path)

type_dict = {
    'caption_type': ['violation', 'status'],
    'violation_type': ['墜落', '機械', '物料', '感電', '防護具', '穿刺', '爆炸', '工作場所', '搬運']
}

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8000, debug=True)