import re
import torch
import numpy as np
from collections import Counter


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train:
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
            padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
              : vocab_size - 4
              ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i + 3 for i, a in enumerate(actions)}
    targets_to_index = {t: i + 3 for i, t in enumerate(targets)}
    actions_to_index["<pad>"] = 0
    actions_to_index["<start>"] = 1
    actions_to_index["<end>"] = 2
    targets_to_index["<pad>"] = 0
    targets_to_index["<start>"] = 1
    targets_to_index["<end>"] = 2
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets


def prefix_match(predicted_labels, gt_labels):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1

    seq_length = len(gt_labels[0])

    for i in range(seq_length):
        tempPred = (predicted_labels[0][i], predicted_labels[1][i])
        tempGT = (gt_labels[0][i], gt_labels[1][i])
        if tempPred != tempGT:
            break

    pm = (1.0 / seq_length) * i

    return pm

# returns the len(longest common substring that starts at the same index because recursion will take too long) / seq
def modLCS(pred, actual):
    # both predicted and actual are 2 x # to predict
    # because I call the shots around here
    lcs = 0
    i = 0
    length = 0
    while i < len(pred[0]):
        if actual[0][i] == 0 and pred[0][i] == 0:  # padding, we don't want to consider this or anything after
            break
        if pred[0][i] == actual[0][i] and pred[1][i] == actual[1][i]:
            length += 1
        else:  # broke streak
            if length > lcs:
                lcs = length
            length = 0

    return (1.0 / (i-1)) * lcs  # do i-1 instead of sequence length because sequence length includes padding, and that's
    # not fair to consider. if padding starts at i, we want to go back one


# returns %  in common (at exact same index)
# ignores padding
def samePlace(pred, actual):
    # again, assuming pred and actual are 2 x seq_len

    seq_len = len(pred[0])
    numSame = 0
    i = 0
    for i in range(seq_len):
        if actual[0][i] == 0 and pred[0][i] == 0:  # padding, nothing after this matters
            break
        if pred[0][i] == actual[0][i] and pred[1][i] == actual[1][i]:
            numSame += 1
    return numSame / (i-1)

# returns % of predictions that at least get the action or target correct
def youTried(pred, actual):
    # same shape assumptions
    seq_len = len(pred[0])
    numClose = 0
    i = 0
    for i in range(seq_len):
        if actual[0][i] == 0 and pred[0][i] == 0:  # padding
            break
        if pred[0][i] == actual[0][i] or pred[1][i] == actual[1][i]:
            numClose += 1
    return numClose / (i-1)
