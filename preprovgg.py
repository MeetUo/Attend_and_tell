import hickle
import tensorflow as tf
from core.vggnet import Vgg19
from scipy import ndimage
from core.untils import *
# vgg model path
vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
# extract conv5_3 feature vectors
vggnet = Vgg19(vgg_model_path)
vggnet.build()
# batch size for extracting feature vectors from vggnet.
batch_size = 80
# because limited by get the feature
feature_size = 10000
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for split in ['train']:
        anno_path = './ch_data/%s/%s.file.names.pkl' % (split, split)
        file_name = load_pickle(anno_path)

        image_path = list(file_name)
        num = int(np.ceil(len(image_path) / feature_size))
        for i in range(0,num):
            save_path = './ch_data/%s/%s.features%d.hkl' % (split, split, i)
            n_examples = feature_size if i < num - 1 else len(image_path) - feature_size * i
            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)
            s1 = i*feature_size
            e1 = i*feature_size+n_examples
            for start, end in zip(range(s1, e1, batch_size),
                                  range(s1+batch_size, e1 + batch_size, batch_size)):
                if (end>len(image_path)) : end = len(image_path)
                image_batch_file = image_path[start:end]
                image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(
                    np.float32)
                # print(start,end)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start-s1:end-s1, :] = feats
                print("Processed %d %s features %d.." % (end, split, i))

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print("Saved %s.." % (save_path))