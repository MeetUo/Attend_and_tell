from collections import Counter
from core.untils import *
import numpy as np
import os
import json
import jieba


def _process_caption_data(caption_file, image_dir, max_length, feature_size):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # data is a list of dictionary which contains 'captions'.
    data = []
    file_name = []
    idc2idm = []
    start = 0
    end = 0

    # caption_s_e'index represent the ith feature should where start and end in caption data
    caption_s_e = []
    for i,item in enumerate(caption_data):
        image_name = item['image_id']
        file_name.append(os.path.join(image_dir, image_name))
        if i%feature_size == 0 and start!=end:
            caption_s_e.append((start,end))
            start = end
        if i%1000==0:
            print("%d/210000 have finished"%i)
        for caption in item['caption']:
            caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
            caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
            caption = jieba.cut(caption)
            caption = " ".join(caption)  # replace multiple spaces
            if len(caption.split(" ")) <= max_length:
                data+=[caption]
                idc2idm.append(i)
                end+=1
    if (start!=end):caption_s_e.append((start,end))

    # print some messages
    print("The number of captions: %d" % len(data))
    print("The captions_s_e:" ,caption_s_e)
    return data,file_name,idc2idm,caption_s_e


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations):
        words = caption.split(' ')
        for w in words:
            counter[w] += 1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print("Max length of caption: ", max_len)
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32)

    for i, caption in enumerate(annotations):
        words = caption.split(" ")
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec)
    print("Finished building caption vectors")
    return captions


def main():
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 20
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # the size of feature split
    feature_size = 10000
    # split train_data to train and test,there is no val data
    caption_file = 'ch_data/caption_train_annotations_20170902.json'

    # about 210000 images and 1050000 captions for train dataset
    train_dataset,file_name,idc2idm,caption_s_e = _process_caption_data(caption_file=caption_file,
                                          image_dir='ch_image/ch_image_train_resized/',
                                          max_length=max_length,feature_size=feature_size)

    split = 'train'
    save_pickle(train_dataset, 'ch_data/train/train.annotations.pkl')
    save_pickle(file_name, './ch_data/%s/%s.file.names.pkl' % (split, split))
    save_pickle(idc2idm, './ch_data/%s/%s.image.idxs.pkl' % (split, split))
    save_pickle(caption_s_e, './ch_data/%s/%s.caption.s&e.pkl' % (split, split))

    print("Start build vocab list")
    annotations = load_pickle('./ch_data/%s/%s.annotations.pkl' % (split, split))

    word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
    save_pickle(word_to_idx, './ch_data/%s/word_to_idx.pkl' % split)

    captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
    save_pickle(captions, './ch_data/%s/%s.captions.pkl' % (split, split))

    print("Finished building %s caption dataset" % split)


if __name__ == "__main__":
    main()
