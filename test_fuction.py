import numpy as np
import jieba
import pickle
from core.untils import *
caption_file = './ch_data/train/word_to_idx.pkl'
word_to_idx = load_pickle('./ch_data/train/word_to_idx.pkl')
print(word_to_idx['奔跑'])