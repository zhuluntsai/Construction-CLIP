import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split

from dataset import caption_dataset, FlickrDataset
from model import EncoderDecoder
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

seed = 567
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def train_transform():
    return torchvision.transforms.Compose([
                    torchvision.transforms.Resize((1024, 1024)),
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

    model_name = f'flickr8k_{data_type}_f{freq_threshold}'
    if embeddings != None:
        model_name += '_embedding'
    logger = SummaryWriter(log_dir=f'log/{model_name}')

    # print('model_name: ', model_name)


    # train_path = 'fengyu/image'
    # train_json = f'fengyu/0_{data_type}.json'
    
    train_path = 'flickr8k/Images'
    train_json = f'flickr8k/captions.txt'

    # train_path = 'chienkuo/image'
    # train_json = 'chienkuo/0.json'

    batch_size = 4

    # train_dataset = caption_dataset(train_path, train_json, transforms=train_transform(), freq_threshold=freq_threshold)
    train_dataset =  FlickrDataset(
        root_dir=train_path,
        caption_file=train_json,
        transform=transform()
    )
    
    
    pad_idx = train_dataset.vocab.stoi["<PAD>"]

    train_size = int(len(train_dataset) * 0.8)
    test_size = len(train_dataset) - train_size
    # train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
    )
                    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_epochs = 25
    start_epochs = 0
    end_epochs = start_epochs + num_epochs

    #Hyperparams
    embed_size=300
    vocab_size = len(train_dataset.vocab)
    attention_dim=256
    encoder_dim=2048
    decoder_dim=512
    learning_rate = 3e-4

    model = EncoderDecoder(
        embed_size=embed_size,
        vocab_size = vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        device=device,
        embeddings=embeddings,
        )

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(start_epochs, end_epochs, 1):
        model.train()

        for i, (imgs, captions) in enumerate(tqdm(train_data_loader)):
            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs, attentions = model(imgs, captions  )
            # print(imgs.shape)
            # print(captions.shape)
            # print(outputs.shape)
            # print(attentions.shape)
            # losses = criterion(outputs, captions.flatten())
            targets = captions[:,1:]
            losses = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

            logger.add_scalar('loss', losses, epoch * len(train_data_loader) + i + 1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 500 == 0:
                print(f'Epoch: {epoch}/{end_epochs}, Batch: {i + 1}/{len(train_data_loader)}, Loss: {losses}')

            # logger.add_scalar('training loss',
            #                     losses,
            #                     epoch * len(train_data_loader) + i + 1)

        # model.eval()
        # correct = 0

        # with torch.no_grad():
        #     for i, (imgs, labels) in enumerate(val_data_loader):
        #         imgs = imgs.to(device)
        #         labels = labels.to(device)
                
        #         outputs = model(imgs)
        #         preds = torch.argmax(outputs, dim=1)
        #         correct += (preds==labels).sum().item()

        # accuracy = correct / len(val_data_loader.dataset)
        # print(f'Epoch: {epoch}/{end_epochs}, Accuracy: {accuracy:.3f}')
        # logger.add_scalar('accuracy',
        #                     accuracy,
        #                     epoch * len(train_data_loader) + i)

        # if previous_accuracy <= accuracy:
        torch.save({'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,}, f'model/{model_name}_latest.pth')
        #     previous_accuracy = accuracy
    
    torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses,}, F'model/{model_name}_final.pth')
    print(data_type, freq_threshold)

if __name__ == "__main__":
    main()