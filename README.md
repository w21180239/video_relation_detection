# pytorch implementation of video captioning

recommend installing pytorch and python packages using Anaconda

## requirements

- cuda
- pytorch 0.4.0
- python3
- ffmpeg (can install using anaconda)

### python packages

- tqdm
- pillow
- pretrainedmodels
- nltk

## Data

MSR-VTT. Test video doesn't have captions, so I spilit train-viedo to train/val/test. Extract and put them in `./data/`
directory

- train-video: [download link](https://drive.google.com/file/d/1Qi6Gn_l93SzrvmKQQu-drI90L-x8B0ly/view?usp=sharing)
- test-video: [download link](https://drive.google.com/file/d/10fPbEhD-ENVQihrRvKFvxcMzkDlhvf4Q/view?usp=sharing)
- json info of
  train-video: [download link](https://drive.google.com/file/d/1LcTtsAvfnHhUfHMiI4YkDgN7lF1-_-m7/view?usp=sharing)
- json info of
  test-video: [download link](https://drive.google.com/file/d/1Kgra0uMKDQssclNZXRLfbj9UQgBv-1YE/view?usp=sharing)

## Options

all default options are defined in opt.py or corresponding code file, change them for your like.

## Acknowledgements

Some code refers
to [ImageCaptioning.pytorch](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

## Data

put kaggle data into data/5242_data and unzip it. But I have done it and preprocess the data. So when you run this
project with the original dataset, you don't need to do that.

## Usage

### (Optional) c3d features

you can use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to
extract features from video.

### Steps

1. preprocess videos and labels

```bash
python prepro_feats.py 

python prepro_vocab.py

python prepro_val_data.py
```

2. Training a model

```bash

python train.py --gpu 0 --epochs 151 --batch_size 128 --checkpoint_path data/save --feats_dir data/feats/resnet152 --model S2VTAttModel  --with_c3d 1 --dim_vid 2048 --max_len 5
```

3. test

   opt_info.json will be in same directory as saved model.

```bash
python eval.py --recover_opt data/save/opt_info.json --saved_model data/save/model_50.pth --batch_size 100 --gpu 0
```

4. predict

```bash
python predict.py --recover_opt data/save/opt_info.json --saved_model data/save/model_50.pth --batch_size 128 --gpu 0
```