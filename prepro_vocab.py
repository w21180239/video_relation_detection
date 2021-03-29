import argparse
import json
import os
from random import shuffle


def main(params):
    object = json.load(open(params['object_json'], 'r'))
    relation = json.load(open(params['relationship_json'], 'r'))

    reverse_object, reverse_relation = {}, {}
    for k, v in object.items():
        reverse_object[v] = k
    for k, v in relation.items():
        reverse_relation[v] = k
    label = json.load(open(params['annotation_json'], 'r'))

    video_caption = {}
    for k, v in label.items():
        sentence = f'{reverse_object[v[0]]} {reverse_relation[v[1]]} {reverse_object[v[2]]}'
        word_list = ['<sos>', reverse_object[v[0]], reverse_relation[v[1]], reverse_object[v[2]], '<eos>']
        video_caption[f'video{int(k)}'] = {'captions': [sentence], 'final_captions': [word_list]}

    # create the vocab
    vocab = [k for k in object.keys()]
    vocab.extend([k for k in relation.keys()])
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'

    train_data_list = [int(i) for i in os.listdir(os.path.join(params['data_dic'], 'train'))]
    shuffle(train_data_list)
    val_data_list, train_data_list = train_data_list[:len(train_data_list) // 5], train_data_list[
                                                                                  len(train_data_list) // 5:]
    test_data_list = [int(i) for i in os.listdir(os.path.join(params['data_dic'], 'test'))]

    out = {}
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    out['videos'] = {'train': train_data_list, 'val': val_data_list,
                     'test': [i + len(train_data_list) + len(val_data_list) for i in test_data_list]}

    json.dump(out, open(params['output_info'], 'w'))
    json.dump(video_caption, open(params['output_label'], 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--object_json', type=str, default='data/5242_data/object1_object2.json',
                        help='object info')
    parser.add_argument('--relationship_json', default='data/5242_data/relationship.json',
                        help='relationship info')
    parser.add_argument('--annotation_json', default='data/5242_data/training_annotation.json', help='training label')
    parser.add_argument('--data_dic', default='data/5242_data',
                        help='training and validate dataset dictionary')

    parser.add_argument('--output_info', default='data/5242_data/info.json',
                        help='word dict and dataset split information')
    parser.add_argument('--output_label', default='data/5242_data/train_val_label.json',
                        help='training and validate dataset label')

    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
