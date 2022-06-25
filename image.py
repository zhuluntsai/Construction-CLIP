# -*- coding:UTF-8 -*-
import fitz, io, os, json
from PIL import Image
from tqdm import tqdm
import docx2txt

import warnings
warnings.filterwarnings("ignore")

# https://www.thepythoncode.com/article/extract-pdf-images-in-python
# https://drive.google.com/drive/u/3/folders/1G7Yjsee0sbhEH7Dfgl2lWmh7KAVCJfKM
# https://github.com/ckiplab/ckip-transformers

# caption_type
v = 'violation'
s = 'status'

def remove_element(list, remove_list):
   return [value for value in list if value not in remove_list]

def convert_report():
    file_names = os.listdir('fengyu/report')
    empty_json = {"images": [], "type": "captions", "annotations": []}

    i = 0
    j = 0

    file_namess = [
        '20200921-邱董至大埔美榮勝廠房-品質及勞安查核報告.pdf', # A
        '20200331-葉主任至福展廠房-安衛走動管理稽查簡報.pdf', # B
        '20201016-李技師至康舒廠房-品質及勞安查核報告.pdf', # C
        '20200306-喬技師至海創三A0南頂版-品質及勞安查核報告.pdf', # D 
    ]

    count = [0, 0, 0, 0, 0]

    for file_name in tqdm(file_names):
        if file_name.endswith('pdf'):
            
            # open the file
            pdf_file = fitz.open(os.path.join('fengyu/report', file_name))
            project_name = file_name.strip().split('-')[-2].split('至')[-1].strip()

            type = ''

            for page_index in range(len(pdf_file)):
                page = pdf_file[page_index]
                content = page.get_text('text')

                if '公司內部文件' in content:
                    type = 'A'
                    count[0] += 1
                    break
                elif '工程說明' in content:
                    type = 'B'
                    count[1] += 1
                    break
                elif '缺失1' in content:
                    type = 'C'
                    count[2] += 1
                    break
                elif '工地現況' in content:
                    type = 'D'
                    count[3] += 1
                    break
                else:
                    type = 'E'

            if type == 'E':
                print(file_name, type)
            
            # iterate over PDF pages
            # page_index start from second page
            for page_index in range(1, len(pdf_file), 1):
                page = pdf_file[page_index]
                image_list = page.getImageList()            

                for image_index, img in enumerate(page.getImageList(), start=1):
                    base_image = pdf_file.extractImage(img[0])
                    image = Image.open(io.BytesIO(base_image["image"]))
                    image_name = f'{file_name[:8]}_{project_name}_{page_index+1}_{image_index}.{base_image["ext"]}'

                    caption = page.get_text('text')
                    if type == 'A':
                        # 公司內部文件，限內部審閱\n查核照片\n查核項目：\n查核項目：\n查核項目：\n查核項目： 屋頂尚未整理\n半邊鷹架扶手先行\n
                        if '缺失改善' in caption:
                            caption_type = v
                        else:
                            caption_type = s
                        caption = caption.split('查核項目：')[-1][:-1].strip().replace('\n', '，')
                        caption = caption.split('缺失改善，')[-1]

                    elif type == 'B':
                        # 二.工程現況\n1. 人員均依規定量測體溫並記錄備查及工地提\n供酒精和之態樣。\n
                        if '缺失改善' in caption:
                            caption_type = v
                        else:
                            caption_type = s
                        caption = caption.split('.')[-1][:-1].strip().replace('\n', '')
                    
                    elif type == 'C':
                        # 說明：柱頭箍筋多數還未調整好，請多補照片，另\n柱牆接合鋼筋務必施作。\n缺失5\n改善照片與說明：\n
                        caption = caption.split('改善照片與說明')[0].split('缺失')[0].split('提醒')[0].split('說明：')[-1].replace('\n', '')
                    
                    elif type == 'D':
                        # 工地現況\nA0南棟一樓整理，作為勞工休息區。\n泥作材料進場。\n
                        if '缺失' in caption:
                            caption_type = v
                        else:
                            caption_type = s
                        caption = caption.split('改善照片與說明')[0].split('缺失')[0].split('提醒')[0].split('說明：')[-1].replace('\n', '')
                        caption = caption.split('工地現況')[-1].replace('\n', '').split('。')
                        caption.insert(0, '0')
                        try:
                            if caption[image_index - 1] != '':
                                caption = caption[image_index - 1]
                            else:
                                caption = caption[1]
                        except:
                            caption = caption[1]

                    else:
                        break
                    caption_original = caption

                    if any(word in caption for word in ['工地名稱', '、安衛', '廠商當月', '豐譽企業團隊', '今日']):
                        # print(caption)
                        break
                    if any(word == caption for word in ['']):
                        # print(caption)
                        break


                    # if any(word in caption for word in ['開口', '護欄', '防墜網', '施工架', '鷹架']):
                    #     print(caption)
                    # else:
                    #     break

                    # caption = caption.replace('Line', '')
                    # caption = caption.replace('line', '')
                    # caption = caption.replace('@', '')
                    # caption = caption.replace('、', '')
                    # caption = caption.split('FL')[-1]

                    # empty = []
                    # pos_sentence_list = []
                    # word_sentence_list = ws([caption])[0]
                    # try:
                    #     pos_sentence_list = pos(word_sentence_list)
                    # except:
                    #     pass

                    # for word, post_list in zip(word_sentence_list, pos_sentence_list):
                    #     p = 1
                    #     for post in post_list:
                    #         if post in ['FW', 'PAUSECATEGORY', 'Nc', 'Ncd', 'COMMACATEGORY']:
                    #             p = 0   
                    #     if p: empty.append(word)

                    # caption = ''.join(empty)
                    # catpion = caption.strip()

                    # caption = caption.replace('層', '')
                    # caption = caption.replace('樓', '')
                    # caption = caption.replace('版', '')

                    # print(caption_original)
                    # print(catpion)
                    # print()

                    annotation_content = {
                        "id": j + 1,
                        "type": type,
                        "report_file_name": file_name,
                        "page": page_index + 1,
                        "caption_type": caption_type,
                        "violation_type": '',
                        "violation_list": '',
                        "original_caption": caption_original,
                        "caption": caption,
                        "file_name": f"fengyu/report_output/{image_name}",
                    }

                    i += 1
                    j += 1

                    if image_index != 1:
                        image.save(open(f"fengyu/report_output/{image_name}", "wb"))
                        empty_json['annotations'].append(annotation_content)

    with open('fengyu/0_all.json', 'w') as outfile:
        json.dump(empty_json, outfile, indent = 2, ensure_ascii = False)

    print(count)

def convert_folder_fenygu():

    root = 'fengyu'
    output_path = 'fengyu/fengyu_month.json'
    folder_path = []
    empty_json = {"type": "captions", "annotations": []}

    for y in range(2021, 2022 + 1, 1):
        for m in range(1, 12 + 1, 1):
            folder_path.append(os.path.join(root, f'{y}年{m:02}月照片'))

    i = 0
    for folder in folder_path:
        try:
            listdir = sorted(os.listdir(folder))
        except:
            continue

        for file_name in listdir:
            annotation_content = {
                "id": i,
                "caption_type": 'violation',
                "violation_type": '',
                "violation_list": '',
                "caption": file_name.split('.')[0],
                "file_name": os.path.join(folder, file_name),
            }
            empty_json['annotations'].append(annotation_content)
            i += 1

    with open(output_path, 'w') as outfile:
        json.dump(empty_json, outfile, indent = 2, ensure_ascii = False)
    
    print(i)

def convert_folder_fenygu_other():

    root = 'fengyu'
    output_path = 'fengyu/fengyu_other.json'
    folder_path = ['其他']
    empty_json = {"type": "captions", "annotations": []}

    i = 0
    for folder in folder_path:
        for file_name in sorted(os.listdir(os.path.join(root, folder))):
            annotation_content = {
                "id": i,
                "caption_type": 'violation',
                "violation_type": '',
                "violation_list": '',
                "caption": '',
                "file_name": os.path.join(folder, file_name),
            }
            empty_json['annotations'].append(annotation_content)
            i += 1

    with open(output_path, 'w') as outfile:
        json.dump(empty_json, outfile, indent = 2, ensure_ascii = False)
    
    print(i)

def convert_folder_reju():

    root = 'reju'
    output_path = 'reju/reju.json'
    folder_path = []
    empty_json = {"type": "captions", "annotations": []}

    for c in ['合格', '不合格']:
        for o in ['開口', '施工架', '安全帽', '其他']:
            folder_path.append(os.path.join(root, f'{c}/{o}'))

    i = 0
    for folder in folder_path:
        try:
            listdir = sorted(os.listdir(folder))
        except:
            continue
        
        if '/合格' in folder:
            caption_type = s
        else:
            caption_type = v

        if '開口' in folder: 
            violation_type = '墜落'
            objects = '開口'
        elif '施工架' in folder: 
            violation_type = '墜落'
            objects = '施工架'
        elif '安全帽' in folder:
            violation_type = '防護具'
            objects = '安全帽'
        else:
            violation_type = ''
            objects = ''

        for file_name in listdir:
            annotation_content = {
                "id": i,
                "caption_type": caption_type,
                "violation_type": violation_type,
                "violation_list": '',
                "caption": '',
                "file_name": os.path.join(folder, file_name),
                "objects": objects
            }
            empty_json['annotations'].append(annotation_content)
            i += 1

    with open(output_path, 'w') as outfile:
        json.dump(empty_json, outfile, indent = 2, ensure_ascii = False)
    
    print(i)

def convert_doc():

    root = 'chienkuo'
    output_path = 'chienkuo/chienkuo.json'
    output_folder = 'output_doc'
    empty_json = {"type": "captions", "annotations": []}
    text_list = []

    id = 0
    for file_name in tqdm(os.listdir(root)):
        if file_name.endswith('docx') and not file_name.startswith('~$'):
            date = file_name[6:12]
            all_text = docx2txt.process(os.path.join(root, file_name), os.path.join(root, output_folder)) 
            all_text = all_text.replace('\n\n\n', '').split('\n')[3:]
            all_text = remove_element(all_text, ['缺失說明', '照片', ''])

            try:
                for i, text in enumerate(all_text):
                    # image index start from 1
                    i += 1 
                    try:
                        os.rename(os.path.join(root, f'{output_folder}/image{i}.jpg'), os.path.join(root, f'{output_folder}/{date}_{i}.jpg'))
                    except:
                        os.rename(os.path.join(root, f'{output_folder}/image{i}.jpeg'), os.path.join(root, f'{output_folder}/{date}_{i}.jpg'))
            except:
                pass

            print(file_name, i, len(all_text))
            if i != len(all_text):
                all_text = docx2txt.process(os.path.join(root, file_name), os.path.join(root, output_folder)) 
                # all_text = all_text.replace('\n\n\n', '').split('外業安全')
                # print(all_text)
                for index in range(i):
                    caption = all_text[2*index] + '，' + all_text[2*index + 1]
            else:
                for i, text in enumerate(all_text):
                    # image index start from 1
                    
                    annotation_content = {
                        "id": id,
                        "report_file_name": file_name,
                        "caption_type": 'violation',
                        "violation_type": '',
                        "violation_list": text,
                        "caption": text,
                        "file_name": f'{date}_{i + 1}.jpg',
                    }
                    id += 1

                    text_list.append(text)
                    empty_json['annotations'].append(annotation_content)

    with open(output_path, 'w') as outfile:
        json.dump(empty_json, outfile, indent = 2, ensure_ascii = False)

    f = open('violation_list.txt', 'w')
    for t in set(sorted(text_list)):
        f.write(t + '\n')
    f.close()

    print(id)


def image_name_correction():
    output_path = 'chienkuo/chienkuo.json'
    all_json = json.load(open(output_path, 'r'))

    for a in all_json['annotations']:
        if '2022' in a['report_file_name']:
            dot = a['file_name'].split('.')

            ext = dot[-1]
            underline = dot[0].split('_')

            file_month = underline[0]
            index = int(underline[-1]) - 1

            a['file_name'] = f'{file_month}_{index}.{ext}'

    with open(output_path, 'w') as outfile:
        json.dump(all_json, outfile, indent = 2, ensure_ascii = False)


def count(output_path):
    all_json = json.load(open(output_path, 'r'))

    count_dict = {
        "caption_type": {
            'violation': 0,
            'status': 0,
        },
        "violation_type": 0,
        "violation_list": 0,
        "caption": 0,
    }

    for k in count_dict.keys():
        if k == 'caption_type':
            for kk in count_dict[k].keys():
                count_dict[k][kk] = sum(a[k] == kk for a in all_json['annotations'])
        else:
            count_dict[k] = sum(a[k] != '' for a in all_json['annotations'])
    
    print('count file: \t\t', output_path)
    print('total annotation: \t', len(all_json['annotations']))
    print(count_dict)

def add_key(output_path):
    all_json = json.load(open(output_path, 'r'))

    for a in all_json['annotations']:
        a['objects'] = ''

    with open(output_path, 'w') as outfile:
        json.dump(all_json, outfile, indent = 2, ensure_ascii = False)

def add_path(output_path):
    all_json = json.load(open(output_path, 'r'))
    # prefix = 'chienkuo/output_doc/'
    prefix = 'fengyu/'

    for a in all_json['annotations']:
        a['file_name'] = prefix + a['file_name'] 

    with open(output_path, 'w') as outfile:
        json.dump(all_json, outfile, indent = 2, ensure_ascii = False)


def combine(path_list):
    output_path = 'all.json'
    empty_json = {"type": "captions", "annotations": []}

    for path in path_list:

        all_json = json.load(open(path, 'r'))
        for a in all_json['annotations']:
            empty_json['annotations'].append(a)

    print(len(empty_json['annotations']))

    with open(output_path, 'w') as outfile:
        json.dump(empty_json, outfile, indent = 2, ensure_ascii = False)

def main():

    output_path_list = [
        'chienkuo/chienkuo.json',
        'reju/reju.json',
        'fengyu/fengyu_month.json',
        'fengyu/fengyu_other.json',
        # 'fengyu/fengyu_report.json',
        'all.json',
    ]

    # convert_report()
    # convert_folder_fenygu()
    # convert_folder_fenygu_other()
    # convert_folder_reju()
    # convert_doc()
    # image_name_correction()

    count(output_path_list[-1])

    # for o in output_path_list:
    #     add_key(o)
    # add_path(output_path_list[3])

    # combine(output_path_list)

if __name__ == '__main__':
    main()