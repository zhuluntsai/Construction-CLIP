import torch
import torch.utils.data
import torchvision
import warnings
warnings.filterwarnings('ignore')
from torch.nn.utils.rnn import pad_sequence

from dataset import caption_dataset, FlickrDataset
from model import EncoderDecoder
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from PIL import Image
import numpy as np
import cv2

def generate_captions(features_tensors, model, device, dataset, font):
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features,vocab=dataset.vocab)
        print(len(alphas))

        caption = ' '.join(caps)
        # show_image(features_tensors[0], font,title=caption)
    
    return caps, alphas

def show_image(img, font, pred_captions, gt_captions, filename):    
    plt.margins(1)
    plt.title(f'pred: {"".join(pred_captions)} \n ground truth: {gt_captions}', fontproperties=font)
    plt.tight_layout()
    plt.imshow(img)
    plt.savefig(filename)


def normalize_list(list_normal):
    max_value = np.max(list_normal)
    min_value = np.min(list_normal)
    list_normal = (list_normal - min_value) / (max_value - min_value)
    return list_normal

def show_attention(img, result, attention_plot, font, filename):    

    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img
    len_result = len(result)

    # fig = plt.figure(figsize=(len_result, len_result/2))
    fig = plt.figure(figsize=(15, 15))
    # for l in range(len_result - 1):
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(7, 7)
        # temp_att = normalize_list(temp_att)
        temp_att = cv2.resize(temp_att, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # ax = fig.add_subplot(len_result/(len_result//2), len_result//2, l+1)
        ax = fig.add_subplot(len_result//2,len_result//2, l+1)
        ax.set_title(result[l], fontproperties=font)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

def train_transform():
    return torchvision.transforms.Compose([
                    torchvision.transforms.Resize((128, 128)),
                    torchvision.transforms.ToTensor(),
                    ])

def transform():
    return torchvision.transforms.Compose([
                    torchvision.transforms.Resize(226),                     
                    torchvision.transforms.RandomCrop(224),                 
                    torchvision.transforms.ToTensor(),                               
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                    ])


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets

def main():
    data_type = 'all'
    # data_type = 'violation'
    # data_type = 'keyword'

    freq_threshold = 5

    # embeddings = torch.load("embeddings.pt")
    embeddings = None

    model_name = f'fengyu_{data_type}_f{freq_threshold}'
    if embeddings != None:
        model_name += '_embedding'

    print('model_name: ', model_name)
    
    # train_path = 'fengyu/image'
    # train_json = f'fengyu/0_{data_type}.json'
    model_name = f'model/{model_name}_final.pth'

    train_path = 'flickr8k/Images'
    train_json = f'flickr8k/captions.txt'

    # train_path = 'chienkuo/image'
    # train_json = 'chienkuo/0.json'
    # model_name = 'chienkuo_final.pth'

    font = font_manager.FontProperties(fname="STHeiti-Medium.ttc")
    # train_dataset = caption_dataset(train_path, train_json, transforms=train_transform(), freq_threshold=freq_threshold)
    train_dataset =  FlickrDataset(
        root_dir=train_path,
        caption_file=train_json,
        transform=transform()
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    embed_size=300
    vocab_size = len(train_dataset.vocab)
    attention_dim=256
    encoder_dim=2048
    decoder_dim=512
    learning_rate = 3e-4

    model = EncoderDecoder(
        embed_size=embed_size,
        vocab_size = len(train_dataset.vocab),
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        device=device,
        embeddings=embeddings,
        )

    model.to(device)
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    caps = []
    pred_caps = []
    
    for index in range(len(train_dataset)):
        # index = 102
        # index = 15
        # index = 6
        target = '3677693858_62f2f3163f'
        # target = '2113530024_5bc6a90e42'
        if target not in train_dataset.imgs.values[index]:
            continue

        try:
            path = train_dataset.coco.loadImgs(index)[0]['file_name']
            gt_captions = train_dataset.coco.anns[index]['caption']
        except:
            path = train_dataset.imgs.values[index]
            gt_captions = train_dataset.captions.values[index]
            caps.append(gt_captions)

        original_image = Image.open(os.path.join(train_dataset.root_dir, path)).convert('RGB')
        transform_image = train_dataset.transform(original_image)

        pred_captions, alphas = generate_captions(transform_image.unsqueeze(0), model, device, train_dataset, font)
        pred_caps.append(' '.join(pred_captions))

    show_image(original_image, font, '\n'.join(pred_caps), '\n'.join(caps), f'output/{index}.png')
    show_attention(transform_image, pred_captions, alphas, font, f'{index}_attention.png')

    print('pred: ', pred_caps)
    print('ground_truth: ', caps) 

    # print(f'pred: {"".join(pred_captions)}')
    # print(f'gt: {gt_captions}')

if __name__ == "__main__":
    main()