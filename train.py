from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.untils import load_coco_data


def main():
    # load train dataset
    data = load_coco_data(data_path='./ch_data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    # val_data = load_coco_data(data_path='./data', split='val')
    val_data = None
    Ch_size = 21
    En_size = 16
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=Ch_size, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=128, update_rule='adam',
                                          learning_rate=0.001, print_every=1000, save_every=5, image_path='./ch_image/',
                                    pretrained_model=None, model_path='ch_model/lstm/', test_model='ch_model/lstm/model-10',
                                     print_bleu=True, log_path='log/')

    solver.train()

if __name__ == "__main__":
    main()