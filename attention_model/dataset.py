import torch
from pycocotools.coco import COCO
from PIL import Image
from collections import Counter
import spacy
import os
import pandas as pd

spacy_zh = spacy.load("zh_core_web_sm")
spacy_en = spacy.load("en_core_web_sm")

# https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/notebook
class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_en.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ] 


class caption_dataset(torch.utils.data.Dataset):

    def __init__(self, root, json_path, transforms=None, freq_threshold=5):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(json_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.vocab = Vocabulary(freq_threshold)

        vocab_list = []
        for i in self.coco.anns:
            vocab_list.append(self.coco.anns[i]['caption'])
        self.vocab.build_vocab(vocab_list)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transforms(img)

        # ann_ids = coco.getAnnIds(imgIds=img_id)
        # coco_annotation = coco.anns[index]['caption']
        caption = coco.anns[img_id]['caption']
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        caption_vec = torch.tensor(caption_vec)
        
        return img, caption_vec

    def __len__(self):
        return len(self.ids)


class FlickrDataset(torch.utils.data.Dataset):
    """
    FlickrDataset
    """
    def __init__(self,root_dir,caption_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        
        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return img, torch.tensor(caption_vec)
