import json

json_path = '../fengyu/0_all.json'

data = json.load(open(json_path, 'r'))
text = ''
for a in data['annotations']:
    text += a['caption']

print(text)

f = open('text.txt', 'w')

f.write

f.write(text)
f.close()