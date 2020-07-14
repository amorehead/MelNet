# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import importlib
# if importlib.util.find_spec('jamo') is None:
#   !pip install jamo
# if importlib.util.find_spec('audiosegment') is None:
#   !pip install audiosegment
# if importlib.util.find_spec('unidecode') is None:
#   !pip install unidecode


# %%
# if importlib.util.find_spec('google') is not None:
#     from google.colab import drive
#     drive.mount('/content/drive')


# %%
# Define imports
import os
import time
import logging
import argparse
import platform

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import traceback
from tqdm import tqdm

import numpy as np

from jamo import h2j, j2h
from jamo.jamo import _jamo_char_to_hcj
import re
import ast
import json
from jamo import hangul_to_jamo, h2j, j2h
import random
import subprocess
import audiosegment

from torch.distributions import Normal

import librosa

import yaml

from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob
from torch.utils.data import Dataset, DataLoader
import string
from unidecode import unidecode  # Added to support LJ_speech
import inflect

import nltk
nltk.download('punkt')

from torch.utils.data import Dataset, DataLoader
import glob


# %%
# Define constants in "constant.py"
# x:y = tiers:(axis should be divisible by)
t_div = {1:1, 2:1, 3:2, 4:2, 5:4, 6:4}
f_div = {1:1, 2:1, 3:2, 4:2, 5:4, 6:4, 7:8}


# %%
# Define constants in "utils.py"
PAD = '_'
EOS = '~'
PUNC = '!$%&*\'()+,-.:;"\\?`/'
SPACE = ' '
NUMBERS = '0123456789'
SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
en_symbols = SYMBOLS + NUMBERS + PAD + EOS + PUNC + SPACE
_symbol_to_id = {s: i for i, s in enumerate(en_symbols)}


# %%
# Define constants from "korean.py"
PAD = '_'
EOS = '~'
PUNC = '!$%&*\'()+,-.:;"\\?`/'
SPACE = ' '

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE
ALL_SYMBOLS = PAD + EOS + VALID_CHARS

char_to_id = {c: i for i, c in enumerate(ALL_SYMBOLS)}
id_to_char = {i: c for i, c in enumerate(ALL_SYMBOLS)}

quote_checker = """([`"'＂“‘])(.+?)([`"'＂”’])"""

num_to_kor = {
        '0': '영',
        '1': '일',
        '2': '이',
        '3': '삼',
        '4': '사',
        '5': '오',
        '6': '육',
        '7': '칠',
        '8': '팔',
        '9': '구',
}

unit_to_kor1 = {
        '%': '퍼센트',
        'cm': '센치미터',
        'mm': '밀리미터',
        'km': '킬로미터',
        'kg': '킬로그람',
}
unit_to_kor2 = {
        'm': '미터',
}

upper_to_kor = {
        'A': '에이',
        'B': '비',
        'C': '씨',
        'D': '디',
        'E': '이',
        'F': '에프',
        'G': '지',
        'H': '에이치',
        'I': '아이',
        'J': '제이',
        'K': '케이',
        'L': '엘',
        'M': '엠',
        'N': '엔',
        'O': '오',
        'P': '피',
        'Q': '큐',
        'R': '알',
        'S': '에스',
        'T': '티',
        'U': '유',
        'V': '브이',
        'W': '더블유',
        'X': '엑스',
        'Y': '와이',
        'Z': '지',
}

number_checker = "([+-]?\d[\d,]*)[\.]?\d*"
count_checker = "(시|명|가지|살|마리|포기|송이|수|톨|통|점|개|벌|척|채|다발|그루|자루|줄|켤레|그릇|잔|마디|상자|사람|곡|병|판)"

num_to_kor1 = [""] + list("일이삼사오육칠팔구")
num_to_kor2 = [""] + list("만억조경해")
num_to_kor3 = [""] + list("십백천")

#count_to_kor1 = [""] + ["하나","둘","셋","넷","다섯","여섯","일곱","여덟","아홉"]
count_to_kor1 = [""] + ["한","두","세","네","다섯","여섯","일곱","여덟","아홉"]

count_tenth_dict = {
        "십": "열",
        "두십": "스물",
        "세십": "서른",
        "네십": "마흔",
        "다섯십": "쉰",
        "여섯십": "예순",
        "일곱십": "일흔",
        "여덟십": "여든",
        "아홉십": "아흔",
}


# %%
# Define symbols from "symbols.py"
# coding: utf-8
'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''

# For english
en_symbols = SYMBOLS + NUMBERS + PAD + EOS + PUNC + SPACE  #<-For deployment(Because korean ALL_SYMBOLS follow this convention)
symbols = ALL_SYMBOLS # for korean

"""
초성과 종성은 같아보이지만, 다른 character이다.

'_~ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ!'(),-.:;? '

'_': 0, '~': 1, 'ᄀ': 2, 'ᄁ': 3, 'ᄂ': 4, 'ᄃ': 5, 'ᄄ': 6, 'ᄅ': 7, 'ᄆ': 8, 'ᄇ': 9, 'ᄈ': 10, 
'ᄉ': 11, 'ᄊ': 12, 'ᄋ': 13, 'ᄌ': 14, 'ᄍ': 15, 'ᄎ': 16, 'ᄏ': 17, 'ᄐ': 18, 'ᄑ': 19, 'ᄒ': 20, 
'ᅡ': 21, 'ᅢ': 22, 'ᅣ': 23, 'ᅤ': 24, 'ᅥ': 25, 'ᅦ': 26, 'ᅧ': 27, 'ᅨ': 28, 'ᅩ': 29, 'ᅪ': 30, 
'ᅫ': 31, 'ᅬ': 32, 'ᅭ': 33, 'ᅮ': 34, 'ᅯ': 35, 'ᅰ': 36, 'ᅱ': 37, 'ᅲ': 38, 'ᅳ': 39, 'ᅴ': 40, 
'ᅵ': 41, 'ᆨ': 42, 'ᆩ': 43, 'ᆪ': 44, 'ᆫ': 45, 'ᆬ': 46, 'ᆭ': 47, 'ᆮ': 48, 'ᆯ': 49, 'ᆰ': 50, 
'ᆱ': 51, 'ᆲ': 52, 'ᆳ': 53, 'ᆴ': 54, 'ᆵ': 55, 'ᆶ': 56, 'ᆷ': 57, 'ᆸ': 58, 'ᆹ': 59, 'ᆺ': 60, 
'ᆻ': 61, 'ᆼ': 62, 'ᆽ': 63, 'ᆾ': 64, 'ᆿ': 65, 'ᇀ': 66, 'ᇁ': 67, 'ᇂ': 68, '!': 69, "'": 70, 
'(': 71, ')': 72, ',': 73, '-': 74, '.': 75, ':': 76, ';': 77, '?': 78, ' ': 79
"""


# %%
# Define constants from "ko_dictionary.py"
# coding: utf-8

etc_dictionary = {
        '2 30대': '이삼십대',
        '20~30대': '이삼십대',
        '20, 30대': '이십대 삼십대',
        '1+1': '원플러스원',
        '3에서 6개월인': '3개월에서 육개월인',
}

english_dictionary = {
        'Devsisters': '데브시스터즈',
        'track': '트랙',

        # krbook
        'LA': '엘에이',
        'LG': '엘지',
        'KOREA': '코리아',
        'JSA': '제이에스에이',
        'PGA': '피지에이',
        'GA': '지에이',
        'idol': '아이돌',
        'KTX': '케이티엑스',
        'AC': '에이씨',
        'DVD': '디비디',
        'US': '유에스',
        'CNN': '씨엔엔',
        'LPGA': '엘피지에이',
        'P': '피',
        'L': '엘',
        'T': '티',
        'B': '비',
        'C': '씨',
        'BIFF': '비아이에프에프',
        'GV': '지비',

        # JTBC
        'IT': '아이티',
        'IQ': '아이큐',
        'JTBC': '제이티비씨',
        'trickle down effect': '트리클 다운 이펙트',
        'trickle up effect': '트리클 업 이펙트',
        'down': '다운',
        'up': '업',
        'FCK': '에프씨케이',
        'AP': '에이피',
        'WHERETHEWILDTHINGSARE': '',
        'Rashomon Effect': '',
        'O': '오',
        'OO': '오오',
        'B': '비',
        'GDP': '지디피',
        'CIPA': '씨아이피에이',
        'YS': '와이에스',
        'Y': '와이',
        'S': '에스',
        'JTBC': '제이티비씨',
        'PC': '피씨',
        'bill': '빌',
        'Halmuny': '하모니', #####
        'X': '엑스',
        'SNS': '에스엔에스',
        'ability': '어빌리티',
        'shy': '',
        'CCTV': '씨씨티비',
        'IT': '아이티',
        'the tenth man': '더 텐쓰 맨', ####
        'L': '엘',
        'PC': '피씨',
        'YSDJJPMB': '', ########
        'Content Attitude Timing': '컨텐트 애티튜드 타이밍',
        'CAT': '캣',
        'IS': '아이에스',
        'SNS': '에스엔에스',
        'K': '케이',
        'Y': '와이',
        'KDI': '케이디아이',
        'DOC': '디오씨',
        'CIA': '씨아이에이',
        'PBS': '피비에스',
        'D': '디',
        'PPropertyPositionPowerPrisonP'
        'S': '에스',
        'francisco': '프란시스코',
        'I': '아이',
        'III': '아이아이', ######
        'No joke': '노 조크',
        'BBK': '비비케이',
        'LA': '엘에이',
        'Don': '',
        't worry be happy': ' 워리 비 해피',
        'NO': '엔오', #####
        'it was our sky': '잇 워즈 아워 스카이',
        'it is our sky': '잇 이즈 아워 스카이', ####
        'NEIS': '엔이아이에스', #####
        'IMF': '아이엠에프',
        'apology': '어폴로지',
        'humble': '험블',
        'M': '엠',
        'Nowhere Man': '노웨어 맨',
        'The Tenth Man': '더 텐쓰 맨',
        'PBS': '피비에스',
        'BBC': '비비씨',
        'MRJ': '엠알제이',
        'CCTV': '씨씨티비',
        'Pick me up': '픽 미 업',
        'DNA': '디엔에이',
        'UN': '유엔',
        'STOP': '스탑', #####
        'PRESS': '프레스', #####
        'not to be': '낫 투비',
        'Denial': '디나이얼',
        'G': '지',
        'IMF': '아이엠에프',
        'GDP': '지디피',
        'JTBC': '제이티비씨',
        'Time flies like an arrow': '타임 플라이즈 라이크 언 애로우',
        'DDT': '디디티',
        'AI': '에이아이',
        'Z': '제트',
        'OECD': '오이씨디',
        'N': '앤',
        'A': '에이',
        'MB': '엠비',
        'EH': '이에이치',
        'IS': '아이에스',
        'TV': '티비',
        'MIT': '엠아이티',
        'KBO': '케이비오',
        'I love America': '아이 러브 아메리카',
        'SF': '에스에프',
        'Q': '큐',
        'KFX': '케이에프엑스',
        'PM': '피엠',
        'Prime Minister': '프라임 미니스터',
        'Swordline': '스워드라인',
        'TBS': '티비에스',
        'DDT': '디디티',
        'CS': '씨에스',
        'Reflecting Absence': '리플렉팅 앱센스',
        'PBS': '피비에스',
        'Drum being beaten by everyone': '드럼 빙 비튼 바이 에브리원',
        'negative pressure': '네거티브 프레셔',
        'F': '에프',
        'KIA': '기아',
        'FTA': '에프티에이',
        'Que sais-je': '',
        'UFC': '유에프씨',
        'P': '피',
        'DJ': '디제이',
        'Chaebol': '채벌',
        'BBC': '비비씨',
        'OECD': '오이씨디',
        'BC': '삐씨',
        'C': '씨',
        'B': '씨',
        'KY': '케이와이',
        'K': '케이',
        'CEO': '씨이오',
        'YH': '와이에치',
        'IS': '아이에스',
        'who are you': '후 얼 유',
        'Y': '와이',
        'The Devils Advocate': '더 데빌즈 어드보카트',
        'YS': '와이에스',
        'so sorry': '쏘 쏘리',
        'Santa': '산타',
        'Big Endian': '빅 엔디안',
        'Small Endian': '스몰 엔디안',
        'Oh Captain My Captain': '오 캡틴 마이 캡틴',
        'AIB': '에이아이비',
        'K': '케이',
        'PBS': '피비에스',
}


# %%
# Define constants from "__init__.py"
# Mappings from symbol to numeric ID and vice versa (Korean characters disabled in 'main()'):
_symbol_to_id = {s: i for i, s in enumerate(en_symbols)}   # 80개
_id_to_symbol = {i: s for i, s in enumerate(en_symbols)}
isEn = True

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

puncuation_table = str.maketrans({key: None for key in string.punctuation})


# %%
# Define constants for "cleaners.py"
# Code based on https://github.com/keithito/tacotron/blob/master/text/cleaners.py
'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).
'''

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


# %%
# Define constants from "en_numbers.py"
_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


# %%
# Define helper functions from "wavloader.py"
def create_dataloader(hp, args, train):
    if args.tts:
        dataset = AudioTextDataset(hp, args, train)
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=train,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=TextCollate()
        )
    else:
        dataset = AudioOnlyDataset(hp, args, train)
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=train,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=AudioCollate()
        )

class AudioOnlyDataset(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.data = hp.data.path
        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

        # this will search all files within hp.data.path
        self.file_list = glob.glob(
            os.path.join(hp.data.path, '**', hp.data.extension),
            recursive=True
        )

        random.seed(123)
        random.shuffle(self.file_list)
        if train:
            self.file_list = self.file_list[:int(0.95 * len(self.file_list))]
        else:
            self.file_list = self.file_list[int(0.95 * len(self.file_list)):]

        self.wavlen = int(hp.audio.sr * hp.audio.duration)
        self.tier = self.args.tier

        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav = read_wav_np(self.file_list[idx], sample_rate=self.hp.audio.sr)
        # wav = cut_wav(self.wavlen, wav)
        mel = self.melgen.get_normalized_mel(wav)
        source, target = self.tierutil.cut_divide_tiers(mel, self.tier)

        return source, target

class AudioTextDataset(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.data = hp.data.path
        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

        # this will search all files within hp.data.path
        self.root_dir = hp.data.path
        self.dataset = []
        if hp.data.name == 'KSS':
            with open(os.path.join(self.root_dir, 'transcript.v.1.3.txt'), 'r') as f:
                lines = f.read().splitlines()
                for line in tqdm(lines):
                    wav_name, _, _, text, length, _ = line.split('|')

                    wav_path = os.path.join(self.root_dir, 'kss', wav_name)
                    duraton = float(length)
                    if duraton < hp.audio.duration:
                        self.dataset.append((wav_path, text))

                # if len(self.dataset) > 100: break
        elif hp.data.name == 'Blizzard':
            with open(os.path.join(self.root_dir, 'prompts.gui'), 'r') as f:
                lines = f.read().splitlines()
                filenames = lines[::3]
                sentences = lines[1::3]
                for filename, sentence in tqdm(zip(filenames, sentences), total=len(filenames)):
                    wav_path = os.path.join(self.root_dir, 'wavn', filename + '.wav')
                    length = get_length(wav_path, hp.audio.sr)
                    if length < hp.audio.duration:
                        self.dataset.append((wav_path, sentence))

        elif hp.data.name == 'Trump':
            with open(os.path.join(self.root_dir, 'prompts.gui'), 'r') as f:
                lines = f.read().splitlines()
                filenames = lines[::3]
                sentences = lines[1::3]
                for filename, sentence in tqdm(zip(filenames, sentences), total=len(filenames)):
                    wav_path = os.path.join(self.root_dir, filename)
                    length = get_length(wav_path, hp.audio.sr)
                    if length < hp.audio.duration:
                        self.dataset.append((wav_path, sentence))
        else:
            raise NotImplementedError

        random.seed(123)
        random.shuffle(self.dataset)
        if train:
            self.dataset = self.dataset[:int(0.95 * len(self.dataset))]
        else:
            self.dataset = self.dataset[int(0.95 * len(self.dataset)):]

        self.wavlen = int(hp.audio.sr * hp.audio.duration)
        self.tier = self.args.tier

        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][1]
        if self.hp.data.name == 'KSS':
            seq = text_to_sequence(text)
        elif self.hp.data.name == 'Blizzard':
            seq = process_blizzard(text)
        elif self.hp.data.name == 'Trump':
            seq = process_trump(text)

        wav = read_wav_np(self.dataset[idx][0], sample_rate=self.hp.audio.sr)
        # wav = cut_wav(self.wavlen, wav)
        mel = self.melgen.get_normalized_mel(wav)
        source, target = self.tierutil.cut_divide_tiers(mel, self.tier)

        return seq, source, target

class TextCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        seq = [torch.from_numpy(x[0]).long() for x in batch]
        text_lengths = torch.LongTensor([x.shape[0] for x in seq])
        # Right zero-pad all one-hot text sequences to max input length
        seq_padded = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)

        audio_lengths = torch.LongTensor([x[1].shape[1] for x in batch])
        source_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x[1].T) for x in batch],
            batch_first=True
        ).transpose(1, 2)
        target_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x[2].T) for x in batch],
            batch_first=True
        ).transpose(1, 2)

        return seq_padded, text_lengths, source_padded, target_padded, audio_lengths

class AudioCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        audio_lengths = torch.LongTensor([x[0].shape[1] for x in batch])
        source_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x[0].T) for x in batch],
            batch_first=True
        ).transpose(1, 2)
        target_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x[1].T) for x in batch],
            batch_first=True
        ).transpose(1, 2)

        return source_padded, target_padded, audio_lengths


# %%
# Define helper functions from "en_numbers.py"
def _remove_commas(m):
  return m.group(1).replace(',', '')


def _expand_decimal_point(m):
  return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
  match = m.group(1)
  parts = match.split('.')
  if len(parts) > 2:
    return match + ' dollars'  # Unexpected format
  dollars = int(parts[0]) if parts[0] else 0
  cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
  if dollars and cents:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
  elif dollars:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    return '%s %s' % (dollars, dollar_unit)
  elif cents:
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s' % (cents, cent_unit)
  else:
    return 'zero dollars'


def _expand_ordinal(m):
  return _inflect.number_to_words(m.group(0))


def _expand_number(m):
  num = int(m.group(0))
  if num > 1000 and num < 3000:
    if num == 2000:
      return 'two thousand'
    elif num > 2000 and num < 2010:
      return 'two thousand ' + _inflect.number_to_words(num % 100)
    elif num % 100 == 0:
      return _inflect.number_to_words(num // 100) + ' hundred'
    else:
      return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
  else:
    return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
  text = re.sub(_comma_number_re, _remove_commas, text)
  text = re.sub(_pounds_re, r'\1 pounds', text)
  text = re.sub(_dollars_re, _expand_dollars, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_ordinal_re, _expand_ordinal, text)
  text = re.sub(_number_re, _expand_number, text)
  return text


# %%
# Define helper functions from "cleaners.py"
def korean_cleaners(text):
    '''Pipeline for Korean text, including number and abbreviation expansion.'''
    text = tokenize(text) # '존경하는' --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']
    return text


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    '''Converts to ascii, existed in keithito but deleted in carpedm20'''
    return unidecode(text)
    

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


# %%
# Define helper functions from "__init__.py"
def convert_to_en_symbols():
    '''Converts built-in korean symbols to english, to be used for english training
    
'''
    global _symbol_to_id, _id_to_symbol, isEn, en_symbols, PAD, EOS, PUNC, SPACE, NUMBERS, SYMBOLS
    if not isEn:
        print(" [!] Converting to english mode")
    PAD = '_'
    EOS = '~'
    PUNC = '!$%&*\'()+,-.:;"\\?`/'
    SPACE = ' '
    NUMBERS = '0123456789'
    SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    en_symbols = SYMBOLS + NUMBERS + PAD + EOS + PUNC + SPACE
    _symbol_to_id = {s: i for i, s in enumerate(en_symbols)}
    _id_to_symbol = {i: s for i, s in enumerate(en_symbols)}
    isEn = True

def remove_puncuations(text):
    return text.translate(puncuation_table)

def text_to_sequence(text, cleaner_names=['korean_cleaners'], as_token=False):    
    return _text_to_sequence(text, cleaner_names, as_token)

def _text_to_sequence(text, cleaner_names, as_token):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

        The text can optionally have ARPAbet sequences enclosed in curly braces embedded
        in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

        Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through

        Returns:
            List of integers corresponding to the symbols in the text
    '''
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # Append EOS token
    sequence.append(_symbol_to_id[EOS])  # [14, 29, 45, 2, 27, 62, 20, 21, 4, 39, 45, 1]

    if as_token:
        return sequence_to_text(sequence, combine_jamo=True)
    else:
        return np.array(sequence, dtype=np.int32)


def sequence_to_text(sequence, skip_eos_and_pad=False, combine_jamo=False):
    '''Converts a sequence of IDs back to a string'''
        
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]

            if not skip_eos_and_pad or s not in [EOS, PAD]:
                result += s

    result = result.replace('}{', ' ')

    if combine_jamo:
        return jamo_to_korean(result)
    else:
        return result



def _clean_text(text, cleaner_names):
    
    for name in cleaner_names:
        from text import cleaners
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text) # '존경하는' --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']

    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


# %%
# Define helper functions in "korean.py"
def is_lead(char):
    return char in JAMO_LEADS

def is_vowel(char):
    return char in JAMO_VOWELS

def is_tail(char):
    return char in JAMO_TAILS

def get_mode(char):
    if is_lead(char):
        return 0
    elif is_vowel(char):
        return 1
    elif is_tail(char):
        return 2
    else:
        return -1

def _get_text_from_candidates(candidates):
    if len(candidates) == 0:
        return ""
    elif len(candidates) == 1:
        return _jamo_char_to_hcj(candidates[0])
    else:
        return j2h(**dict(zip(["lead", "vowel", "tail"], candidates)))

def jamo_to_korean(text):
    text = h2j(text)

    idx = 0
    new_text = ""
    candidates = []

    while True:
        if idx >= len(text):
            new_text += _get_text_from_candidates(candidates)
            break

        char = text[idx]
        mode = get_mode(char)

        if mode == 0:
            new_text += _get_text_from_candidates(candidates)
            candidates = [char]
        elif mode == -1:
            new_text += _get_text_from_candidates(candidates)
            new_text += char
            candidates = []
        else:
            candidates.append(char)

        idx += 1
    return new_text

def compare_sentence_with_jamo(text1, text2):
    return h2j(text1) != h2j(text2)

def tokenize(text, as_id=False):
    # jamo package에 있는 hangul_to_jamo를 이용하여 한글 string을 초성/중성/종성으로 나눈다.
    text = normalize(text)
    tokens = list(hangul_to_jamo(text)) # '존경하는'  --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']

    if as_id:
        return [char_to_id[token] for token in tokens] + [char_to_id[EOS]]
    else:
        return [token for token in tokens] + [EOS]

def tokenizer_fn(iterator):
    return (token for x in iterator for token in tokenize(x, as_id=False))

def normalize(text):
    text = text.strip()

    text = re.sub('\(\d+일\)', '', text)
    text = re.sub('\([⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+\)', '', text)

    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    text = re.sub('[a-zA-Z]+', normalize_upper, text)

    text = normalize_quote(text)
    text = normalize_number(text)

    return text

def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile('|'.join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    else:
        return text

def normalize_english(text):
    def fn(m):
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        else:
            return word

    text = re.sub("([A-Za-z]+)", fn, text)
    return text

def normalize_upper(text):
    text = text.group(0)

    if all([char.isupper() for char in text]):
        return "".join(upper_to_kor[char] for char in text)
    else:
        return text

def normalize_quote(text):
    def fn(found_text):
        from nltk import sent_tokenize # NLTK doesn't along with multiprocessing

        found_text = found_text.group()
        unquoted_text = found_text[1:-1]

        sentences = sent_tokenize(unquoted_text)
        return " ".join(["'{}'".format(sent) for sent in sentences])

    return re.sub(quote_checker, fn, text)

def normalize_number(text):
    text = normalize_with_dictionary(text, unit_to_kor1)
    text = normalize_with_dictionary(text, unit_to_kor2)
    text = re.sub(number_checker + count_checker,
            lambda x: number_to_korean(x, True), text)
    text = re.sub(number_checker,
            lambda x: number_to_korean(x, False), text)
    return text

def number_to_korean(num_str, is_count=False):
    if is_count:
        num_str, unit_str = num_str.group(1), num_str.group(2)
    else:
        num_str, unit_str = num_str.group(), ""
    
    num_str = num_str.replace(',', '')
    num = ast.literal_eval(num_str)

    if num == 0:
        return "영"

    check_float = num_str.split('.')
    if len(check_float) == 2:
        digit_str, float_str = check_float
    elif len(check_float) >= 3:
        raise Exception(" [!] Wrong number format")
    else:
        digit_str, float_str = check_float[0], None

    if is_count and float_str is not None:
        raise Exception(" [!] `is_count` and float number does not fit each other")

    digit = int(digit_str)

    if digit_str.startswith("-"):
        digit, digit_str = abs(digit), str(abs(digit))

    kor = ""
    size = len(str(digit))
    tmp = []

    for i, v in enumerate(digit_str, start=1):
        v = int(v)

        if v != 0:
            if is_count:
                tmp += count_to_kor1[v]
            else:
                tmp += num_to_kor1[v]

            tmp += num_to_kor3[(size - i) % 4]

        if (size - i) % 4 == 0 and len(tmp) != 0:
            kor += "".join(tmp)
            tmp = []
            kor += num_to_kor2[int((size - i) / 4)]

    if is_count:
        if kor.startswith("한") and len(kor) > 1:
            kor = kor[1:]

        if any(word in kor for word in count_tenth_dict):
            kor = re.sub(
                    '|'.join(count_tenth_dict.keys()),
                    lambda x: count_tenth_dict[x.group()], kor)

    if not is_count and kor.startswith("일") and len(kor) > 1:
        kor = kor[1:]

    if float_str is not None:
        kor += "쩜 "
        kor += re.sub('\d', lambda x: num_to_kor[x.group()], float_str)

    if num_str.startswith("+"):
        kor = "플러스 " + kor
    elif num_str.startswith("-"):
        kor = "마이너스 " + kor

    return kor + unit_str


# %%
# Define helper functions in "utils.py"
def get_length(wavpath, sample_rate):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    return audio.duration_seconds

def process_blizzard(text: str):
    text = text.replace('@ ', '').replace('# ', '').replace('| ', '') + EOS
    seq = [_symbol_to_id[c] for c in text]
    return np.array(seq, dtype=np.int32)

def process_trump(text: str):
    text = text.replace('@ ', '').replace('# ', '').replace('| ', '') + EOS
    seq = [_symbol_to_id[c] for c in text]
    return np.array(seq, dtype=np.int32)

def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def read_wav_np(wavpath, sample_rate):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    wav = audio.to_numpy_array()
    
    if len(wav.shape) == 2:
        wav = wav.T.flatten()
    
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    
    wav = wav.astype(np.float32)
    return wav


def cut_wav(L, wav):
    samples = len(wav)
    if samples < L:
        wav = np.pad(wav, (0, L - samples), 'constant', constant_values=0.0)
    else:
        start = random.randint(0, samples - L)
        wav = wav[start:start + L]

    return wav


def norm_wav(wav):
    assert isinstance(wav, np.ndarray) and len(wav.shape)==1, 'Wav file should be 1D numpy array'
    return wav / np.max( np.abs(wav) )


def trim_wav(wav, threshold=0.01):
    assert isinstance(wav, np.ndarray) and len(wav.shape)==1, 'Wav file should be 1D numpy array'
    cut = np.where((abs(wav)>threshold))[0]
    wav = wav[cut[0]:(cut[-1]+1)]
    return wav


# %%
# Define helper functions of "gmm.py"
def get_pi_indices(pi):
    cumsum = torch.cumsum(pi.cpu(), dim=-1)
    rand = torch.rand(pi.shape[:-1] + (1,))
    indices = (cumsum < rand).sum(dim=-1)
    return indices.flatten().detach().numpy()

def sample_gmm(mu, std, pi):
    std = std.exp()
    pi = pi.softmax(dim=-1)
    indices = get_pi_indices(pi)
    mu = mu.reshape(-1, mu.shape[-1])
    mu = mu[np.arange(mu.shape[0]), indices].reshape(std.shape[:-1])
    std = std.reshape(-1, std.shape[-1])
    std = std[np.arange(std.shape[0]), indices].reshape(mu.shape)
    return torch.normal(mu, std).reshape_as(mu).clamp(0.0, 1.0).to(mu.device)


# %%
# Define train function of "train.py"
def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    if args.tts:
        model = TTS(
            hp=hp,
            freq=hp.audio.n_mels // f_div[hp.model.tier+1] * f_div[args.tier],
            layers=hp.model.layers[args.tier-1]
        )
    else:
        model = Tier(
            hp=hp,
            freq=hp.audio.n_mels // f_div[hp.model.tier+1] * f_div[args.tier],
            layers=hp.model.layers[args.tier-1],
            tierN=args.tier
        )
    model = nn.DataParallel(model).cuda()
    melgen = MelGen(hp)
    tierutil = TierUtil(hp)
    criterion = GMMLoss()

    if hp.train.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=hp.train.rmsprop.lr, 
            momentum=hp.train.rmsprop.momentum
        )
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=hp.train.adam.lr
        )
    elif hp.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=hp.train.sgd.lr
        )
    else:
        raise Exception("%s optimizer not supported yet" % hp.train.optimizer)

    # githash = get_commit_hash()

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        print("Resuming from checkpoint: %s" % chkpt_path)
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        # if githash != checkpoint['githash']:
        #     logger.warning("Code might be different: git hash is different.")
        #     logger.warning("%s -> %s" % (checkpoint['githash'], githash))

        # githash = checkpoint['githash']
    else:
        print("Starting new training run.")
        logger.info("Starting new training run.")

    # use this only if input size is always consistent.
    # torch.backends.cudnn.benchmark = True
    try:
        model.train()
        optimizer.zero_grad()
        loss_sum = 0
        for epoch in itertools.count(init_epoch + 1):
            loader = tqdm(trainloader, desc='Train data loader', dynamic_ncols=True)
            for input_tuple in loader:
                if args.tts:
                    seq, text_lengths, source, target, audio_lengths = input_tuple
                    mu, std, pi, _ = model(
                        source.cuda(non_blocking=True),
                        seq.cuda(non_blocking=True),
                        text_lengths.cuda(non_blocking=True),
                        audio_lengths.cuda(non_blocking=True)
                    )
                else:
                    source, target, audio_lengths = input_tuple
                    mu, std, pi = model(
                        source.cuda(non_blocking=True),
                        audio_lengths.cuda(non_blocking=True)
                    )
                loss = criterion(
                    target.cuda(non_blocking=True),
                    mu, std, pi,
                    audio_lengths.cuda(non_blocking=True)
                )
                step += 1
                (loss / hp.train.update_interval).backward()
                loss_sum += loss.item() / hp.train.update_interval

                if step % hp.train.update_interval == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if step % hp.log.summary_interval == 0:
                        writer.log_training(loss_sum, step)
                        loader.set_description("Loss %.04f at step %d" % (loss_sum, step))
                    loss_sum = 0

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.04f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

            save_path = os.path.join(pt_dir, '%s_tier%d_%03d.pt') % (
                args.name, 
                # githash,
                args.tier, 
                epoch
            ))

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
                # 'githash': githash,
            }, save_path)

            print("Saved checkpoint to: %s" % save_path)
            logger.info("Saved checkpoint to: %s" % save_path)

            validate(args, model, melgen, tierutil, testloader, criterion, writer, step)

    except Exception as e:
        print("Exiting due to exception: %s" % e)
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()


# %%
# Define validate function from "validation.py"
def validate(args, model, melgen, tierutil, testloader, criterion, writer, step):
    model.eval()
    # torch.backends.cudnn.benchmark = False

    test_loss = []
    loader = tqdm(testloader, desc='Testing is in progress', dynamic_ncols=True)
    with torch.no_grad():
        for input_tuple in loader:
            if args.tts:
                seq, text_lengths, source, target, audio_lengths = input_tuple
                mu, std, pi, alignment = model(
                    source.cuda(non_blocking=True),
                    seq.cuda(non_blocking=True),
                    text_lengths.cuda(non_blocking=True),
                    audio_lengths.cuda(non_blocking=True)
                )
            else:
                source, target, audio_lengths = input_tuple
                mu, std, pi = model(
                    source.cuda(non_blocking=True),
                    audio_lengths.cuda(non_blocking=True)
                )
            loss = criterion(
                target.cuda(non_blocking=True),
                mu, std, pi,
                audio_lengths.cuda(non_blocking=True)
            )
            test_loss.append(loss)

        test_loss = sum(test_loss) / len(test_loss)
        audio_length = audio_lengths[0].item()
        source = source[0].cpu().detach().numpy()[:, :audio_length]
        target = target[0].cpu().detach().numpy()[:, :audio_length]
        result = sample_gmm(mu[0], std[0], pi[0]).cpu().detach().numpy()[:, :audio_length]
        if args.tts:
            alignment = alignment[0].cpu().detach().numpy()[:, :audio_length]
        else:
            alignment = None
        writer.log_validation(test_loss, source, target, result, alignment, step)

    model.train()
    # torch.backends.cudnn.benchmark = True


# %%
# Define helper functions from "hparams.py"
def load_hparam_str(hp_str):
    path = os.path.join('temp-restore.yaml')
    with open(path, 'w') as f:
        f.write(hp_str)
    ret = HParam(path)
    os.remove(path)
    return ret


def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


# %%
# Define helper functions from "plotting.py"
def fig2np(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.transpose(2, 0, 1)
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


# %%
# Define classes of "hparams.py"
class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class HParam(Dotdict):

    def __init__(self, file):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
            
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__


# %%
# Define MyWriter class from "writer.py"
class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)
        # self.add_histogram('mu', mu, step)
        # self.add_histogram('std', std, step)
        # self.add_histogram('std_exp', std.exp(), step)
        # self.add_histogram('pi', pi, step)
        # self.add_histogram('pi_softmax', pi.softmax(dim=3), step)

    def log_validation(self, test_loss, source, target, result, alignment, step):
        self.add_scalar('test_loss', test_loss, step)
        self.add_image('input', plot_spectrogram_to_numpy(source), step)
        self.add_image('target', plot_spectrogram_to_numpy(target), step)
        self.add_image('result', plot_spectrogram_to_numpy(result), step)
        self.add_image('diff', plot_spectrogram_to_numpy(target - result), step)
        if alignment is not None:
            self.add_image('alignment', plot_spectrogram_to_numpy(alignment.T), step)

    def log_sample(self, step):
        raise NotImplementedError


# %%
# Define tier class of "tier.py"
class Tier(nn.Module):
    def __init__(self, hp, freq, layers, tierN):
        super(Tier, self).__init__()
        num_hidden = hp.model.hidden
        self.hp = hp
        self.tierN = tierN

        if(tierN == 1):
            self.W_t_0 = nn.Linear(1, num_hidden)
            self.W_f_0 = nn.Linear(1, num_hidden)
            self.W_c_0 = nn.Linear(freq, num_hidden)
            self.layers = nn.ModuleList([
                DelayedRNN(hp) for _ in range(layers)
            ])
        else:
            self.W_t = nn.Linear(1, num_hidden)
            self.layers = nn.ModuleList([
                UpsampleRNN(hp) for _ in range(layers)
            ])

        # Gaussian Mixture Model: eq. (2)
        self.K = hp.model.gmm
        self.pi_softmax = nn.Softmax(dim=3)

        # map output to produce GMM parameter eq. (10)
        self.W_theta = nn.Linear(num_hidden, 3*self.K)

    def forward(self, x, audio_lengths):
        # x: [B, M, T] / B=batch, M=mel, T=time
        if self.tierN == 1:
            h_t = self.W_t_0(F.pad(x, [1, -1]).unsqueeze(-1))
            h_f = self.W_f_0(F.pad(x, [0, 0, 1, -1]).unsqueeze(-1))
            h_c = self.W_c_0(F.pad(x, [1, -1]).transpose(1, 2))
            for layer in self.layers:
                h_t, h_f, h_c = layer(h_t, h_f, h_c, audio_lengths)

            # h_t, h_f: [B, M, T, D] / D=num_hidden
            # h_c: [B, T, D]
        else:
            h_f = self.W_t(x.unsqueeze(-1))
            for layer in self.layers:
                h_f = layer(h_f, audio_lengths)

        theta_hat = self.W_theta(h_f)

        mu = theta_hat[..., :self.K] # eq. (3)
        std = theta_hat[..., self.K:2*self.K]
        pi = theta_hat[..., 2*self.K:]

        return mu, std, pi


# %%
# Define delayedRNN class of "rnn.py"
class DelayedRNN(nn.Module):
    def __init__(self, hp):
        super(DelayedRNN, self).__init__()
        self.num_hidden = hp.model.hidden

        self.t_delay_RNN_x = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True
        )
        self.t_delay_RNN_yz = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True,
            bidirectional=True
        )

        # use central stack only at initial tier
        self.c_RNN = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True
        )
        self.f_delay_RNN = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True
        )

        self.W_t = nn.Linear(3*self.num_hidden, self.num_hidden)
        self.W_c = nn.Linear(self.num_hidden, self.num_hidden)
        self.W_f = nn.Linear(self.num_hidden, self.num_hidden)
   
    def flatten_rnn(self):
        self.t_delay_RNN_x.flatten_parameters()
        self.t_delay_RNN_yz.flatten_parameters()
        self.c_RNN.flatten_parameters()
        self.f_delay_RNN.flatten_parameters()

    def forward(self, input_h_t, input_h_f, input_h_c, audio_lengths):
      
        self.flatten_rnn()
        # input_h_t, input_h_f: [B, M, T, D]
        # input_h_c: [B, T, D]
        B, M, T, D = input_h_t.size()

        ####### time-delayed stack #######
        # Fig. 2(a)-1 can be parallelized by viewing each horizontal line as batch
        h_t_x_temp = input_h_t.view(-1, T, D)
        h_t_x_packed = nn.utils.rnn.pack_padded_sequence(
            h_t_x_temp,
            audio_lengths.unsqueeze(1).repeat(1, M).reshape(-1),
            batch_first=True,
            enforce_sorted=False
        )
        h_t_x, _ = self.t_delay_RNN_x(h_t_x_packed)
        h_t_x, _ = nn.utils.rnn.pad_packed_sequence(
            h_t_x,
            batch_first=True,
            total_length=T
        )
        h_t_x = h_t_x.view(B, M, T, D)

        # Fig. 2(a)-2,3 can be parallelized by viewing each vertical line as batch,
        # using bi-directional version of GRU
        h_t_yz_temp = input_h_t.transpose(1, 2).contiguous() # [B, T, M, D]
        h_t_yz_temp = h_t_yz_temp.view(-1, M, D)
        h_t_yz, _ = self.t_delay_RNN_yz(h_t_yz_temp)
        h_t_yz = h_t_yz.view(B, T, M, 2*D)
        h_t_yz = h_t_yz.transpose(1, 2)

        h_t_concat = torch.cat((h_t_x, h_t_yz), dim=3)
        output_h_t = input_h_t + self.W_t(h_t_concat) # residual connection, eq. (6)

        ####### centralized stack #######
        h_c_temp = nn.utils.rnn.pack_padded_sequence(
            input_h_c,
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        h_c_temp, _ = self.c_RNN(h_c_temp)
        h_c_temp, _ = nn.utils.rnn.pad_packed_sequence(
            h_c_temp,
            batch_first=True,
            total_length=T
        )
            
        output_h_c = input_h_c + self.W_c(h_c_temp) # residual connection, eq. (11)
        h_c_expanded = output_h_c.unsqueeze(1)

        ####### frequency-delayed stack #######
        h_f_sum = input_h_f + output_h_t + h_c_expanded
        h_f_sum = h_f_sum.transpose(1, 2).contiguous() # [B, T, M, D]
        h_f_sum = h_f_sum.view(-1, M, D)

        h_f_temp, _ = self.f_delay_RNN(h_f_sum)
        h_f_temp = h_f_temp.view(B, T, M, D)
        h_f_temp = h_f_temp.transpose(1, 2) # [B, M, T, D]
        
        output_h_f = input_h_f + self.W_f(h_f_temp) # residual connection, eq. (8)

        return output_h_t, output_h_f, output_h_c


# %%
# Define upsampleRNN of "upsample.py"
class UpsampleRNN(nn.Module):
    def __init__(self, hp):
        super(UpsampleRNN, self).__init__()
        self.num_hidden = hp.model.hidden

        self.rnn_x = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.rnn_y = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.W = nn.Linear(4 * self.num_hidden, self.num_hidden)

    def flatten_parameters(self):
        self.rnn_x.flatten_parameters()
        self.rnn_y.flatten_parameters()

    def forward(self, inp, audio_lengths):
        self.flatten_parameters()
        
        B, M, T, D = inp.size()

        inp_temp = inp.view(-1, T, D)
        inp_temp = nn.utils.rnn.pack_padded_sequence(
            inp_temp,
            audio_lengths.unsqueeze(1).repeat(1, M).reshape(-1),
            batch_first=True,
            enforce_sorted=False
        )
        x, _ = self.rnn_x(inp_temp)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x,
            batch_first=True,
            total_length=T
        )
        x = x.view(B, M, T, 2 * D)

        y, _ = self.rnn_y(inp.transpose(1, 2).contiguous().view(-1, M, D))
        y = y.view(B, T, M, 2 * D).transpose(1, 2).contiguous()

        z = torch.cat([x, y], dim=-1)

        output = inp + self.W(z)

        return output


# %%
# Define attention from "tts.py"
class Attention(nn.Module):
    def __init__(self, hp):
        super(Attention, self).__init__()
        self.M = hp.model.gmm
        self.rnn_cell = nn.LSTMCell(
            input_size=2*hp.model.hidden,
            hidden_size=hp.model.hidden
        )
        self.W_g = nn.Linear(hp.model.hidden, 3*self.M)
        
    def attention(self, h_i, memory, ksi):
        phi_hat = self.W_g(h_i)

        ksi = ksi + torch.exp(phi_hat[:, :self.M])
        beta = torch.exp(phi_hat[:, self.M:2*self.M])
        alpha = F.softmax(phi_hat[:, 2*self.M:3*self.M], dim=-1)
        
        u = memory.new_tensor(np.arange(memory.size(1)), dtype=torch.float)
        u_R = u + 1.5
        u_L = u + 0.5
        
        term1 = torch.sum(
            alpha.unsqueeze(-1) * torch.sigmoid(
                (u_R - ksi.unsqueeze(-1)) / beta.unsqueeze(-1)
            ),
            keepdim=True,
            dim=1
        )
        
        term2 = torch.sum(
            alpha.unsqueeze(-1) * torch.sigmoid(
                (u_L - ksi.unsqueeze(-1)) / beta.unsqueeze(-1)
            ),
            keepdim=True,
            dim=1
        )
        
        weights = term1 - term2
        
        context = torch.bmm(weights, memory)
        
        termination = 1 - term1.squeeze(1)

        return context, weights, termination, ksi # (B, 1, D), (B, 1, T), (B, T)

    
    
    def forward(self, input_h_c, memory):
        B, T, D = input_h_c.size()
        
        context = input_h_c.new_zeros(B, D)
        h_i, c_i  = input_h_c.new_zeros(B, D), input_h_c.new_zeros(B, D)
        ksi = input_h_c.new_zeros(B, self.M)
        
        contexts, weights = [], []
        for i in range(T):
            x = torch.cat([input_h_c[:, i], context.squeeze(1)], dim=-1)
            h_i, c_i = self.rnn_cell(x, (h_i, c_i))
            context, weight, termination, ksi = self.attention(h_i, memory, ksi)
            
            contexts.append(context)
            weights.append(weight)
            
        contexts = torch.cat(contexts, dim=1) + input_h_c
        alignment = torch.cat(weights, dim=1)
        # termination = torch.gather(termination, 1, (input_lengths-1).unsqueeze(-1)) # 4

        return contexts, alignment#, termination



class TTS(nn.Module):
    def __init__(self, hp, freq, layers):
        super(TTS, self).__init__()
        self.hp = hp

        self.W_t_0 = nn.Linear(1, hp.model.hidden)
        self.W_f_0 = nn.Linear(1, hp.model.hidden)
        self.W_c_0 = nn.Linear(freq, hp.model.hidden)
        
        self.layers = nn.ModuleList([DelayedRNN(hp) for _ in range(layers)])

        # Gaussian Mixture Model: eq. (2)
        self.K = hp.model.gmm

        # map output to produce GMM parameter eq. (10)
        self.W_theta = nn.Linear(hp.model.hidden, 3*self.K)

        if self.hp.data.name == 'KSS':
            self.embedding_text = nn.Embedding(len(symbols), hp.model.hidden)
        elif self.hp.data.name == 'Blizzard':
            self.embedding_text = nn.Embedding(len(en_symbols), hp.model.hidden)
        elif self.hp.data.name == 'Trump':
            self.embedding_text = nn.Embedding(len(en_symbols), hp.model.hidden)
        else:
            raise NotImplementedError

        self.text_lstm = nn.LSTM(
            input_size=hp.model.hidden,
            hidden_size=hp.model.hidden//2, 
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = Attention(hp)

    def text_encode(self, text, text_lengths):
        total_length = text.size(1)
        embed = self.embedding_text(text)
        packed = nn.utils.rnn.pack_padded_sequence(
            embed,
            text_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        memory, _ = self.text_lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            memory,
            batch_first=True,
            total_length=total_length
        )
        return unpacked
        
    def forward(self, x, text, text_lengths, audio_lengths):
        # Extract memory
        memory = self.text_encode(text, text_lengths)
        
        # x: [B, M, T] / B=batch, M=mel, T=time
        h_t = self.W_t_0(F.pad(x, [1, -1]).unsqueeze(-1))
        h_f = self.W_f_0(F.pad(x, [0, 0, 1, -1]).unsqueeze(-1))
        h_c = self.W_c_0(F.pad(x, [1, -1]).transpose(1, 2))
        
        # h_t, h_f: [B, M, T, D] / h_c: [B, T, D]
        for i, layer in enumerate(self.layers):
            if i != (len(self.layers)//2):
                h_t, h_f, h_c = layer(h_t, h_f, h_c, audio_lengths)
                
            else:
                h_c, alignment = self.attention(h_c, memory)
                h_t, h_f, h_c = layer(h_t, h_f, h_c, audio_lengths)

        theta_hat = self.W_theta(h_f)

        mu = theta_hat[..., :self.K] # eq. (3)
        std = theta_hat[..., self.K:2*self.K] # eq. (4)
        pi = theta_hat[..., 2*self.K:] # eq. (5)
            
        return mu, std, pi, alignment


# %%
# Define GMMLoss from "loss.py"
class GMMLoss(nn.Module):
    def __init__(self):
        super(GMMLoss, self).__init__()

    def forward(self, x, mu, std, pi, audio_lengths):
        x = nn.utils.rnn.pack_padded_sequence(
            x.unsqueeze(-1).transpose(1, 2),
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        ).data
        mu = nn.utils.rnn.pack_padded_sequence(
            mu.transpose(1, 2),
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        ).data
        std = nn.utils.rnn.pack_padded_sequence(
            std.transpose(1, 2),
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        ).data
        pi = nn.utils.rnn.pack_padded_sequence(
            pi.transpose(1, 2),
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        ).data
        log_prob = Normal(loc=mu, scale=std.exp()).log_prob(x)
        log_distrib = log_prob + F.log_softmax(pi, dim=-1)
        loss = -torch.logsumexp(log_distrib, dim=-1).mean()
        return loss


# %%
# Define MelGen from "audio.py"
class MelGen():
    def __init__(self, hp):
        self.hp = hp

    def get_normalized_mel(self, x):
        x = librosa.feature.melspectrogram(
            y=x,
            sr=self.hp.audio.sr,
            n_fft=self.hp.audio.win_length,
            hop_length=self.hp.audio.hop_length,
            win_length=self.hp.audio.win_length,
            n_mels=self.hp.audio.n_mels
        )
        x = self.pre_spec(x)
        return x

    def pre_spec(self, x):
        return self.normalize(librosa.power_to_db(x) - self.hp.audio.ref_level_db)

    def post_spec(self, x):
        return librosa.db_to_power(self.denormalize(x) + self.hp.audio.ref_level_db)

    def normalize(self, x):
        return np.clip(x / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, x):
        return (np.clip(x, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db


# %%
# Define TierUtil from "tierutil.py"
class TierUtil():
    def __init__(self, hp):
        self.hp = hp
        self.n_mels = hp.audio.n_mels

        self.f_div = f_div[hp.model.tier]
        self.t_div = t_div[hp.model.tier]

        # when we perform stft, the number of time frames we get is:
        # self.T = int(hp.audio.sr * hp.audio.duration) // hp.audio.hop_length + 1
        # 10*22050 // 256 + 1 = 862 (blizzard)
        # 6*22050 // 256 + 1 = 517 (maestro)
        # 6*16000 // 180 + 1 = 534 (voxceleb2)
        # 10*16000 // 180 + 1 = 889 (tedlium3)        

    def cut_divide_tiers(self, x, tierNo):
        x = x[:, :x.shape[-1] - x.shape[-1] % self.t_div]
        M, T = x.shape
        assert M % self.f_div == 0,             'freq(mel) dimension should be divisible by %d, got %d.'             % (self.f_div, M)
        assert T % self.t_div == 0,             'time dimension should be divisible by %d, got %d.'             % (self.t_div, T)

        tiers = list()
        for i in range(self.hp.model.tier, max(1, tierNo-1), -1):
            if i % 2 == 0: # make consistent with utils/constant.py
                tiers.append(x[1::2, :])
                x = x[::2, :]
            else:
                tiers.append(x[:, 1::2])
                x = x[:, ::2]
        tiers.append(x)

        # return source, target
        if tierNo == 1:
            return tiers[-1], tiers[-1].copy()
        else:
            return tiers[-1], tiers[-2]

    def interleave(self, x, y, tier):
        '''
            implements eq. (25)
            x: x^{<g}
            y: x^{g}
            tier: g+1
        '''
        assert x.size() == y.size(),             'two inputs for interleave should be identical: got %s, %s' % (x.size(), y.size())

        B, M, T = x.size()
        if tier % 2 == 0:
            temp = x.new_zeros(B, M, 2 * T)
            temp[:, :, 0::2] = x
            temp[:, :, 1::2] = y
        else:
            temp = x.new_zeros(B, 2 * M, T)
            temp[:, 0::2, :] = x
            temp[:, 1::2, :] = y

        return temp


# %%
# Define main function of "trainer.py"
if __name__ == '__main__':
    convert_to_en_symbols()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    parser.add_argument('-t', '--tier', type=int, required=True,
                        help="Number of tier to train")
    parser.add_argument('-b', '--batch_size', type=int, required=True,
                        help="Batch size")
    parser.add_argument('-s', '--tts', type=bool, default=False, required=False,
                        help="TTS")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())
    if platform.system() == 'Windows':
        hp.train.num_workers = 0

    pt_dir = os.path.join(hp.log.chkpt_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    if not os.path.isdir(hp.log.log_dir):
        os.mkdir(hp.log.log_dir)
    if not os.path.isdir(hp.log.chkpt_dir):
        os.mkdir(hp.log.chkpt_dir)
    if not os.path.isdir(pt_dir):
        os.mkdir(pt_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    writer = MyWriter(hp, log_dir)

    assert hp.data.path != '',         'hp.data.path cannot be empty: please fill out your dataset\'s path in configuration yaml file.'
    trainloader = create_dataloader(hp, args, train=True)
    testloader = create_dataloader(hp, args, train=False)

    train(args, pt_dir, args.checkpoint_path, trainloader, testloader, writer, logger, hp, hp_str)

