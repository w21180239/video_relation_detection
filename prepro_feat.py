import argparse
import glob
import os

import numpy as np
import pretrainedmodels
import torch
from pretrainedmodels import utils
from torch import nn
from tqdm import tqdm

C, H, W = 3, 224, 224


def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))

    train_val_frame_path_list = os.listdir(os.path.join(params['video_path'] + '/train'))
    test_frame_path_list = os.listdir(os.path.join(params['video_path'] + '/test'))
    for frame_path in tqdm(train_val_frame_path_list, desc='processing train_val video data'):
        video_id = int(frame_path)
        outfile = os.path.join(dir_fc, f'video{video_id}.npy')
        if os.path.exists(outfile):
            continue

        image_list = sorted(glob.glob(os.path.join(params['video_path'] + '/train/' + frame_path, '*.jpg')))
        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))
        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img
        with torch.no_grad():
            if params['gpu'] != '-1':
                images = images.cuda()
            fc_feats = model(images).squeeze()

        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        np.save(outfile, img_feats)

    for frame_path in tqdm(test_frame_path_list, desc='processing test video data'):
        video_id = int(frame_path) + len(train_val_frame_path_list)
        outfile = os.path.join(dir_fc, f'video{video_id}.npy')
        if os.path.exists(outfile):
            continue

        image_list = sorted(glob.glob(os.path.join(params['video_path'] + '/test/' + frame_path, '*.jpg')))
        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))
        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img
        with torch.no_grad():
            if params['gpu'] != '-1':
                images = images.cuda()
            fc_feats = model(images).squeeze()

        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        np.save(outfile, img_feats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, -1 represent using CPU')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/resnet152', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=40,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/5242_data', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet152',
                        help='the CNN model you want to use to extract_feats')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    model, load_image_fn = None, None
    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))
        exit(-1)

    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)

    if params['gpu'] != '-1':
        model = model.cuda()
    extract_feats(params, model, load_image_fn)
