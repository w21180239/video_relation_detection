import json
import os

import torch
import torch.optim as optim
from ml_metrics import mapk
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

import auto_tune_opt as opts
import misc.utils as utils
from dataloader import VideoDataset
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel


def val_map5(model, val_data, crit, opt):
    fc_feats = val_data['fc_feats']
    labels = val_data['labels']
    masks = val_data['masks']
    model.eval()

    if opt["gpu"] != '-1':
        torch.cuda.synchronize()
        fc_feats = fc_feats.cuda()
        labels = labels.cuda()
        masks = masks.cuda()

    with torch.no_grad():
        _, _, _, all_seq_preds = model(
            fc_feats, mode='inference', opt=opt)
        seq_probs, _, _, _ = model(fc_feats, labels, 'train')
    loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
    val_score = sum(
        [mapk(list(labels[:, i + 1].unsqueeze(1).cpu()), list(all_seq_preds[:, i, :].cpu()), 5)
         for i in range(3)]) / 3

    model.train()
    return loss, val_score


def prepare_work():
    opt = opts.parse_opt()
    opt = vars(opt)
    if opt["gpu"] != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    return opt


def train(train_loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    model.train()
    for data in train_loader:
        fc_feats = data['fc_feats']
        labels = data['labels']
        masks = data['masks']

        if opt["gpu"] != '-1':
            torch.cuda.synchronize()
            fc_feats = fc_feats.cuda()
            labels = labels.cuda()
            masks = masks.cuda()

        optimizer.zero_grad()
        seq_probs, _, _, _ = model(fc_feats, labels, 'train')
        loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])

        loss.backward()
        clip_grad_value_(model.parameters(), opt['grad_clip'])
        optimizer.step()
    lr_scheduler.step()


def one_round_train(config):
    global opt, train_dataset, train_dataloader, val_data

    model = None
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            config["dim_hidden"],
            config["dim_word"],
            opt['dim_vid'] + (opt['c3d_feat_dim'] if opt['with_c3d'] else 0),
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=config["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"] + (opt['c3d_feat_dim'] if opt['with_c3d'] else 0),
            config["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=config["rnn_dropout_p"],
        )
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            config["dim_hidden"],
            config["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=config["rnn_dropout_p"],
            bidirectional=opt["bidirectional"],
            using_gpu=(opt["gpu"] != '-1'))
        model = S2VTAttModel(encoder, decoder)
    else:
        print('invalid model name')
        exit(-1)
    if opt["gpu"] != '-1':
        model = model.cuda()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    for i in range(200):
        train(train_dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)
        val_loss, val_score = val_map5(model, val_data, crit, opt)
        tune.report(mean_accuracy=val_score)


if __name__ == '__main__':
    search_space = {
        "dim_hidden": tune.grid_search([32, 64, 128, 256, 512]),
        'dim_word': tune.grid_search([32, 64, 128, 256, 512]),
        'rnn_dropout_p': tune.uniform(0.4, 0.7),
        "learning_rate": tune.uniform(5e-4, 2e-3),
        'weight_decay': tune.uniform(5e-4, 2e-3),
    }
    opt = prepare_work()
    train_dataset = VideoDataset(opt, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True)
    val_dataset = VideoDataset(opt, 'val')
    val_dataloader = DataLoader(val_dataset, batch_size=opt["batch_size"], shuffle=False)
    val_data = None
    for data in val_dataloader:
        val_data = data
    opt["vocab_size"] = train_dataset.get_vocab_size()

    analysis = tune.run(one_round_train,
                        num_samples=500,
                        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
                        resources_per_trial={"gpu": 1},
                        config=search_space,
                        local_dir='D:\\ray_result',
                        )
    dfs = analysis.trial_dataframes

    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)
