from core.untils import *
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from core.vggnet import Vgg19
from scipy import ndimage


def _process_test_data(image_dir):
    list = os.listdir(image_dir)
    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    id = 0;
    for filename in list:
        annotation = {}
        annotation['file_name'] = os.path.join(image_dir, filename)
        annotation['image_id'] = id
        data += [annotation]
        id += 1

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    return caption_data


def _build_image_idxs(annotations):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = image_id
    return image_idxs


def main():
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 20
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1

    # about 500 images and 2500 captions
    test_dataset = _process_test_data(image_dir='image/XingBi_image_resized/')

    print('Finished processing caption data')

    split = 'X_test'
    save_pickle(test_dataset, 'data/X_test/X_test.annotations.pkl')

    annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))

    file_names = np.asarray(annotations['file_name'])
    save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))

    image_idxs = _build_image_idxs(annotations)
    save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))

    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
    # extract conv5_3 feature vectors
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    # batch size for extracting feature vectors from vggnet.
    batch_size = 80
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        anno_path = './data/%s/%s.annotations.pkl' % (split, split)
        annotations = load_pickle(anno_path)

        image_path = list(annotations['file_name'].unique())
        save_path = './data/%s/%s.features.hkl' % (split, split)

        n_examples = len(image_path)
        all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)
        for start, end in zip(range(0, n_examples, batch_size),
                              range(batch_size, n_examples + batch_size, batch_size)):
            if (end > len(image_path)): end = len(image_path)
            image_batch_file = image_path[start:end]
            image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(
                np.float32)
            # print(start,end)
            feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
            all_feats[start:end, :] = feats
            print("Processed %d %s features.." % (end, split))

        # use hickle to save huge feature vectors
        hickle.dump(all_feats, save_path)
        print("Saved %s.." % (save_path))


if __name__ == "__main__":
    main()
