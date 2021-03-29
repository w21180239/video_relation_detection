import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

import misc.utils as utils
import opts
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from ml_metrics import mapk


def val_map5(model, val_data,crit):
    fc_feats = val_data['fc_feats']
    labels = val_data['labels']
    masks = val_data['masks']

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
    return loss,val_score


def train(train_loader, val_dataloader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    model.train()
    # model = nn.DataParallel(model)
    for epoch in range(opt["epochs"]):
        lr_scheduler.step()

        iteration = 0
        # If start self crit training
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        # batch size must > 117
        val_data = None
        for data in val_dataloader:
            val_data = data
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
            if not sc_flag:
                seq_probs, _, _, _ = model(fc_feats, labels, 'train')
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            else:
                seq_probs, seq_preds = model(
                    fc_feats, mode='inference', opt=opt)
                reward = get_self_critical_reward(model, fc_feats, data,
                                                  seq_preds)
                print(reward.shape)
                loss = rl_crit(seq_probs, seq_preds,
                               torch.from_numpy(reward).float().cuda())

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            if opt["gpu"] != '-1':
                torch.cuda.synchronize()
            iteration += 1

            if not sc_flag:
                print("iter %d (epoch %d), train_loss = %.6f" %
                      (iteration, epoch, train_loss), end='')
            else:
                print("iter %d (epoch %d), avg_reward = %.6f" %
                      (iteration, epoch, np.mean(reward[:, 0])), end='')

        val_loss,val_score = val_map5(model, val_data,crit)
        print(f' val_loss:{val_loss} val_score:{val_score}')

        if epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))


def main(opt):
    train_dataset = VideoDataset(opt, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True)
    val_dataset = VideoDataset(opt, 'val')
    val_dataloader = DataLoader(val_dataset, batch_size=opt["batch_size"], shuffle=False)

    opt["vocab_size"] = train_dataset.get_vocab_size()
    model = None
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'] + (opt['c3d_feat_dim'] if opt['with_c3d'] else 0),
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"] + (opt['c3d_feat_dim'] if opt['with_c3d'] else 0),
            opt["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"])
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
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(train_dataloader, val_dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)


if __name__ == '__main__':
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
    main(opt)
