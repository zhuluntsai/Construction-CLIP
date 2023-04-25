import torch
import TransformerMMExplainability.CLIP.clip as clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
import warnings
warnings.simplefilter("ignore", UserWarning)

start_layer =  -1
start_layer_text =  -1

def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
        # calculate index of last layer
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]


    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1:
        # calculate index of last layer
        start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    
    text_relevance = R_text

    return text_relevance, image_relevance

def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image);
    axs[0].axis('off');

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis);
    axs[1].axis('off');

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def show_heatmap_on_text(text, text_encoding, R_text):
    _tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()

    text_list = [t for t in text]
    text_tokens=[_tokenizer.encode(t) for t in text_list]

    score = []
    score_list = []
    i = 0
    for s in text_scores:
        score.append(s)

        if len(score) == len(text_tokens[i]):
            score_list.append(sum(score).item())
            score = []
            i += 1

    # print('encode: ', len(text_tokens), text_tokens)
    # print('score: ', len(text_scores), text_scores)
    # print('score list: ', len(score_list), score_list)
    # print('text: ', len(text_list), text_list)

    vis_data_records = [visualization.VisualizationDataRecord(score_list,0,0,0,0,0,text,1)]

    html = visualization.visualize_text(vis_data_records).data
    with open('test.html', 'w') as f:
        f.write(html)

def get_clip_attention_model(model_path, device):
    print('get clip attention model')
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    with open(model_path, 'rb') as opened_file: 
        model.load_state_dict(torch.load(opened_file, map_location="cpu"))
    
    return model, preprocess

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # model_path = 'CLIP/models/clip_comb2_0_comb9_6_cap2_5.pt'
    model_path = 'CLIP/models/clip__comb9_999.pt'
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    img_path = 'fengyu/2022年02月照片/筏基孔未設置覆蓋，應圈圍三角錐及連桿警示-2.jpg'
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    # texts = ["缺失 現況"]
    texts = ["開口未設安全護欄"]
    text = clip.tokenize(texts).to(device)

    R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
    batch_size = text.shape[0]
    for i in range(batch_size):
        show_heatmap_on_text(texts[i], text[i], R_text[i])
        show_image_relevance(R_image[i], img, orig_image=Image.open(img_path))
        plt.savefig('test1.png')

if __name__ == '__main__':
    main()