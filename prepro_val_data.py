import argparse
import json
import re


def build_vocab(vids, params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    for vid, caps in vids.items():
        for cap in caps['captions']:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')
    for vid, caps in vids.items():
        caps = caps['captions']
        vids[vid]['final_captions'] = []
        for cap in caps:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            caption = [
                          '<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
            vids[vid]['final_captions'].append(caption)
    return vocab


def main(params):
    object = json.load(open(params['object_json'], 'r'))
    relation = json.load(open(params['relationship_json'], 'r'))

    reverse_object, reverse_relation = {}, {}
    for k, v in object.items():
        reverse_object[v] = k
    for k, v in relation.items():
        reverse_relation[v] = k
    label = json.load(open(params['annotation_json'], 'r'))

    re = {}
    for k, v in label.items():
        re[f'video{int(k)}'] = [{'image_id': f'video{int(k)}', 'cap_id': 0,
                                 'caption': f'{reverse_object[v[0]]} {reverse_relation[v[1]]} {reverse_object[v[2]]}'}]

    json.dump(re, open(params['output_json'], 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--object_json', type=str, default='data/5242_data/object1_object2.json',
                        help='object info')
    parser.add_argument('--relationship_json', default='data/5242_data/relationship.json',
                        help='relationship info')
    parser.add_argument('--annotation_json', default='data/5242_data/training_annotation.json', help='training label')
    parser.add_argument('--output_json', default='data/5242_data/val_info.json',
                        help='info that used in eval')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
