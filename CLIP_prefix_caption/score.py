from nltk.translate import bleu_score
from tqdm import tqdm
import json, os
import numpy as np
from rouge import Rouge
from ckiptagger import WS, data_utils

def bleu():
    data = json.load(open('output_log.json'))
    progress = tqdm(total=len(data['caption']), desc='testing')

    score_list = []
    for d in data['caption']:
        hypothesis = list(d['prediction'])
        reference = [list(d['caption'])]
        
        score = bleu_score.sentence_bleu(reference, hypothesis, smoothing_function=bleu_score.SmoothingFunction().method1)
        score_list.append(score)
        progress.set_postfix({
                    'bleu score': np.mean(score_list),
                    })
        progress.update()

    progress.close()
    print(np.mean(score_list))

def rouge():
    data_dir = ("ckiptagger/data")
    ws = WS(data_dir)



def main():
    bleu()
    rouge()

if __name__ == '__main__':
    main()