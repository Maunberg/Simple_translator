from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.metrics import classification_report, f1_score
import torch.utils.data as data_utils
import torch.optim as optim
import os
import joblib
from catboost import CatBoostClassifier


class Tokeniser():
    def __init__(self, config={}):
        self.vocab_idtotoken = {}
        self.vocab_tokentoid = {}

    def generate_vocab(self, text, create=True, way='vocab.json', use_saved=True):
        if os.path.exists(way) and use_saved:
            self.load_vocab(way)
        special_tokens = ['<EOS>', '<BOS>', '<UNK>']
        # Добавляем языковые токены
        lang_tokens = ['<EN>', '<RU>', '<FR>', '<DE>', '<ES>', '<IT>', '<PT>', '<TR>', '<JP>', '<EO>']
        special_tokens.extend(lang_tokens)
        alpha_en = 'qazxswedcvfrtgbnhyujmkiolp'
        num = '0123456789'
        alpha_ru = 'йфячыцувсмакепитнргоьблшщдюжзхэъ'
        tokens = sorted(set(text+alpha_ru+alpha_ru.upper()+num+alpha_en+alpha_en.upper()))
        tokens_intext = ['##'+ i for i in tokens]
        tokens.extend(tokens_intext)
        tokens.extend(special_tokens)
        tokens.extend(self.get_tokens(2, text, 2000))
        tokens.extend(self.get_tokens(3, text, 1000))
        tokens.extend(self.get_tokens(4, text, 800))
        tokens.extend(self.get_tokens(5, text, 600))
        tokens.extend(self.get_tokens(6, text, 500))
        tokens.extend(self.get_tokens(7, text, 500))
        tokens.extend(self.get_tokens(8, text, 500))
        tokens.extend(self.get_tokens(9, text, 500))
        for id in range(len(tokens)):
            self.vocab_idtotoken[id] = tokens[id]
            self.vocab_tokentoid[tokens[id]] = id
        if create:
            with open(way, 'w') as f:
                json.dump({
                    'vocab_idtotoken':self.vocab_idtotoken,
                    'vocab_tokentoid':self.vocab_tokentoid
                          }, f)
            with open(way) as f:
                data = json.load(f)
            self.vocab_idtotoken = data['vocab_idtotoken']
            self.vocab_tokentoid = data['vocab_tokentoid']

    def __len__(self):
        return len(self.vocab_idtotoken)

    def get_tokens(self, length, text, count_times=30):
        text_to_tokens = re.sub('[^\w@] ', '', text).replace('BOS', '').replace('EOS', '')
        sumbols_count = {}
        for word in tqdm(text_to_tokens.split(' '), desc=f'Generate {length} sumbols vocab'):
            for id in range(len(word)//length):
                token = word[length*id:length*(id+1)]
                if length*id != 0:
                    token = '##'+token
                if token in sumbols_count:
                    sumbols_count[token] += 1
                else:
                    sumbols_count[token] = 1
        sumbols_count = sorted(sumbols_count.items(), key=lambda x: x[1], reverse=True)
        sumbols_count = [i[0] for i in sumbols_count if i[1]>=count_times]
        return sumbols_count
        
    def tokenize(self, text):
        tokenized_text = []
        words = text.split(' ')
        for word in words:
            if not word:
                continue
            word_sub = word
            word_result = []
            while word.replace(' ', '') != '':
                replace_token = '_'
                if replace_token in word:
                    replacevocab = '_*-/=+-—~Æ¯‘„ÎŸ›$#|ˆ{'
                    for i in replacevocab:
                        if i not in word:
                            replace_token = i
                            break
                start = min(9, len(word))
                found = False
                for count in range(start, 0, -1):
                    for i in range(len(word) - count + 1):
                        token = word[i:i + count]
                        if token == ' ':
                            continue
                        id_real = word_sub.find(token)
                        if id_real != 0:
                            token_tosearch = '##'+token
                        else:
                            token_tosearch = token
                        token_id = self.vocab_tokentoid.get(token_tosearch, '')
                        if isinstance(token_id, str) and token_id.isdigit():
                            token_id = int(token_id)
                        if isinstance(token_id, int):
                            word_result.append([int(token_id), id_real])
                            # print(f'Token is |{token}| in the |{word}| from {i} to {i + count}')
                            # print(f'Sub word is |{word_sub}| start token with {id_real}')
                            word_sub = word_sub.replace(token, replace_token*len(token), 1)
                            word = word[:i] + ' ' + word[i + count:]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    token_id = self.vocab_tokentoid.get('<UNK>', None)
                    if token_id is not None:
                        id_real = word_sub.find(token)
                        word_result.append(int(token_id))
                        word_sub = word_sub.replace(token, replace_token*len(token), 1)
                        word = word[1:]
                    else:
                        raise ValueError(f"Неизвестный токен: {token}")
            word_result = [i[0] for i in sorted(word_result, key=lambda x: x[1])]
            tokenized_text.extend(word_result)
        return tokenized_text

    def detokenize(self, tokenized_text):
        text = []
        for id in tqdm(tokenized_text, desc='Detokenising data'):
            token = self.vocab_idtotoken[str(id)]
            text.append(token)
        return text

    def totext(self, tokenized_text):
        text = ''
        tokens = self.detokenize(tokenized_text)
        now = tokens[0]
        for id in range(1, len(tokens)):
            next = tokens[id]
            if '##' in next:
                text+=now.replace('##', '')
            else:
                text+=now.replace('##', '')+' '
            now = next
        text += now.replace('##', '')
        return text

    def load_vocab(self, way):
        with open(way) as f:
            data = json.load(f)
        if len(data['vocab_idtotoken']) == len(data['vocab_tokentoid']):
            self.vocab_idtotoken = data['vocab_idtotoken']
            self.vocab_tokentoid = data['vocab_tokentoid']
    
    def get_lang_token(self, lang_code):
        """Получение языкового токена по коду языка"""
        lang_mapping = {
            'eng': '<EN>',
            'rus': '<RU>', 
            'fra': '<FR>',
            'deu': '<DE>',
            'spa': '<ES>',
            'ita': '<IT>',
            'por': '<PT>',
            'tur': '<TR>',
            'jpn': '<JP>',
            'epo': '<EO>'
        }
        return lang_mapping.get(lang_code, '')
    
    def get_lang_token_id(self, lang_code):
        """Получение ID языкового токена по коду языка"""
        lang_token = self.get_lang_token(lang_code)
        if lang_token:
            return self.vocab_tokentoid.get(lang_token)
        return None
