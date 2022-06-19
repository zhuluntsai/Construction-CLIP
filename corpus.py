#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tempfile import tempdir
import requests
from bs4 import BeautifulSoup
import spacy, os, re, fitz, io
from tqdm import tqdm
from PIL import Image


pref = 'https://zh.wikipedia.org/zh-tw'
title = []
num_layer = 2
spacy_zh = spacy.load("zh_core_web_sm")

def tokenize(text):
    exclude_punctuation = ['、', '，', '，，', ',', '。', '？', '！', '(', ')', '（', '）', '「', '」', '：', '；', '[', ']', '《', '》', '@', ':', '.', '/', '〈', '〉', '_', '#', '~', '-', '─', '┌', '│', '┤', '┐', '└', '┘', '┼', '├', '─', '、', '──', '├─', '││', '─┼', '┼─', '┤│', '│├', '─┬', '一', '○', '○○', '○○○']
    token = [token.text.lower() for token in spacy_zh.tokenizer(text)]
    token = [each_token for each_token in token if re.search("[a-zA-Z]", each_token) == None and not each_token.isdigit() and ' ' not in each_token]
    token = [each_token for each_token in token if all(exclude not in each_token for exclude in exclude_punctuation)]
    token = ' '.join(token)
    return token

def crawler(url_list, text):
    exclude_list = ['Category', 'Special', 'Portal', 'Help', 'index.php', 'wikidata']

    new_url_list = []
    for url in tqdm(url_list):
        response = requests.get(url=url)
        soup = BeautifulSoup(response.content, 'html.parser')

        title.append(soup.find_all('h1')[0].text)

        for paragraph in soup.find_all(['p']):
            
            # text
            new_text = paragraph.text.replace('\n', '')
            text += tokenize(new_text)

            # url
            content = paragraph.find_all(['a'], href=True, class_=False)
            for element in content:
                url = pref + element['href'].replace('/wiki', '')
                if all(str not in url for str in exclude_list) and 'cite_note' not in url:
                    # print(url)
                    new_url_list.append(url)

    return new_url_list, text

def safety_report(text):
    file_names = os.listdir('fengyu/report')

    i = 0

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
                    break
                elif '工程說明' in content:
                    type = 'B'
                    break
                elif '缺失1' in content:
                    type = 'C'
                    break
                elif '工地現況' in content:
                    type = 'D'
                    break
                else:
                    type = 'E'
            
            # iterate over PDF pages
            # page_index start from second page
            for page_index in range(1, len(pdf_file), 1):
                page = pdf_file[page_index]

                for image_index, _ in enumerate(page.getImageList(), start=1):

                    caption = page.get_text('text')
                    if type == 'A':
                        # 公司內部文件，限內部審閱\n查核照片\n查核項目：\n查核項目：\n查核項目：\n查核項目： 屋頂尚未整理\n半邊鷹架扶手先行\n
                        caption = caption.split('查核項目：')[-1][:-1].strip().replace('\n', '，')
                        caption = caption.split('缺失改善，')[-1]
                    elif type == 'B':
                        # 二.工程現況\n1. 人員均依規定量測體溫並記錄備查及工地提\n供酒精和之態樣。\n
                        caption = caption.split('.')[-1][:-1].strip().replace('\n', '')
                    elif type == 'C':
                        # 說明：柱頭箍筋多數還未調整好，請多補照片，另\n柱牆接合鋼筋務必施作。\n缺失5\n改善照片與說明：\n
                        caption = caption.split('改善照片與說明')[0].split('缺失')[0].split('提醒')[0].split('說明：')[-1].replace('\n', '')
                    elif type == 'D':
                        # 工地現況\nA0南棟一樓整理，作為勞工休息區。\n泥作材料進場。\n
                        # if '缺失' in caption:
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
                    # else:
                    #     break

                    text += tokenize(caption)
    return text

def regulation(text):
    file_name = '職業安全衛生設施規則.pdf'

    i = 0

    # open the file
    pdf_file = fitz.open(os.path.join(file_name))

    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        content = page.get_text('text').replace('\n', '')
        
        temp = ''
        content = content.split('、')
        for c in content:
            temp += c[:-1]

        text += tokenize(temp)

    return text

def main():
    text = ''
    length = [0]

    # safety report
    text += safety_report(text)
    length.append(len(text.split(' ')))

    # sefety regulation
    text += regulation(text)
    length.append(len(text.split(' ')))

    # wiki
    url_list = ['https://zh.wikipedia.org/zh-tw/%E5%9C%9F%E6%9C%A8%E5%B7%A5%E7%A8%8B',
            'https://zh.wikipedia.org/wiki/%E5%AE%89%E5%85%A8%E5%B7%A5%E7%A8%8B',
            'https://zh.wikipedia.org/wiki/%E5%A1%94%E5%BC%8F%E8%B5%B7%E9%87%8D%E6%A9%9F',
            'https://zh.wikipedia.org/wiki/%E7%BB%93%E6%9E%84%E5%B7%A5%E7%A8%8B', 
            'https://zh.wikipedia.org/wiki/%E8%81%B7%E6%A5%AD%E5%AE%89%E5%85%A8%E5%81%A5%E5%BA%B7',
            ]
    for _ in range(num_layer):
        url_list, text = crawler(url_list, text)
        print(len(url_list))
        print(title)
    length.append(len(text.split(' ')))

    length = [length[i + 1] - length[i] for i in range(len(length) - 1)]
    print('number of tokens: ', length)
    print('number of titles: ', len(title))

    path = 'output.txt'
    f = open(path, 'w')
    f.write(text)
    f.close()

if __name__ == '__main__':
    main()