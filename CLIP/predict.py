import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load("ViT-B/32", device=device)

model_path = 'models/clip_latest.pt'
with open(model_path, 'rb') as opened_file: 
    model.load_state_dict(torch.load(opened_file, map_location="cpu"))

image_path_list = [
    # '../chienkuo/output_doc/202109_1.jpg', # scaffold
    # '../chienkuo/output_doc/202104_4.jpg', # stair
    # '../chienkuo/output_doc/202107_10.jpg', # gas cylindar
    '../reju/不合格/施工架/e0c9f160-6e01-4c92-9584-293ac69f4342.jpg',
    '../reju/不合格/安全帽/421cf39f-acc6-4306-8a39-5cdf5a66d61d.jpg',
    '../reju/不合格/其他/無交通指揮人員及指揮手-缺.jpg',
]
image_list = []
original_image_list = []
for image_path in image_path_list:
    image = Image.open(image_path)
    original_image_list.append(image)
    image_list.append(preprocess(image))


image = torch.tensor(np.stack(image_list)).to(device)

# text = clip.tokenize(["梯井間未設護欄", "防護具未配戴使用"]).to(device)
# text = clip.tokenize(["缺失", "現況"]).to(device)
# original_text = ["墜落", "防護具", "工作場所"]
original_text = ["violation", "status"]
text = clip.tokenize(original_text).to(device)

with torch.no_grad():
    image_features = model.encode_image(image).float()
    text_features = model.encode_text(text).float()
    
    logits_per_image, logits_per_text = model(image, text)
    similarity = logits_per_image.softmax(dim=-1).cpu().numpy()
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    # similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

print(similarity)

font = font_manager.FontProperties(fname="STHeiti-Medium.ttc")
count = len(original_text)
plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)
# plt.colorbar()
plt.yticks(range(count), original_text, fontsize=18, fontproperties=font)
plt.xticks([])
for i, image in enumerate(original_image_list):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12, fontproperties=font)


plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity between text and image features", size=20)
plt.show()