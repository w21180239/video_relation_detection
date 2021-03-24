import torch
import torch.nn as nn


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


def decode_index_into_final_answer(ix_to_word, object_ix, relation_ix, seq):
    submit_re = []
    visualize_re = {}
    for k in range(seq.shape[0]):
        left_obj, right_obj, relation = [], [], []
        vi_left,vi_re,vi_right = [],[],[]
        for i in range(seq.shape[2]):
            word = ix_to_word[str(seq[k, 0, i].item())]
            if len(left_obj) < 5 and word in object_ix:
                left_obj += [object_ix[word]]
                vi_left+=[word]

        for i in range(seq.shape[2]):
            word = ix_to_word[str(seq[k, 1, i].item())]
            if len(relation) < 5 and word in relation_ix:
                relation += [relation_ix[word]]
                vi_re+=[word]

        for i in range(seq.shape[2]):
            word = ix_to_word[str(seq[k, 2, i].item())]
            if len(right_obj) < 5 and word in object_ix:
                right_obj += [object_ix[word]]
                vi_right+=[word]

        submit_re += [list2str(left_obj), list2str(relation), list2str(right_obj)]
        visualize_re[k] = {'left_object':vi_left,'relation':vi_re,'right_object':vi_right}

    return submit_re,visualize_re

def list2str(arr:[int]) -> str:
    re = ''
    for num in arr:
        re += f'{num} '
    return re[:-1]

class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(),
                          mask[:, :-1]], 1).contiguous().view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output
