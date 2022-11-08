import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
from sklearn.metrics import accuracy_score
import json
import numpy as np

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)

import model as md


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    file = open("../CSCI499_NaturalLanguageforInteractiveAI/hw3/lang_to_sem_data.json")
    jsonLang = json.load(file)
    trainUntoken = jsonLang["train"]
    valUntoken = jsonLang["valid_seen"]

    # table for inputs
    v2i, i2v, maxLen = build_tokenizer_table(trainUntoken)
    #table for outputs
    a2i, i2a, t2i, i2t = build_output_tables(trainUntoken)

    trainEncoded = list()
    trainOut = list()
    for series in trainUntoken:  # each set of commands/labels
        commandTemp = list()
        commandTemp.append(v2i["<start>"])
        actionTemp = list()
        actionTemp.append(v2i["<start>"])
        targetTemp = list()
        targetTemp.append(v2i["<start>"])
        for example in series:  # each pair of command and labels
            temp = preprocess_string(example[0])
            encodingTemp = list()
            wordList = temp.split()
            for word in wordList:
                word = word.lower()
                if len(word) > 0:
                    if word in v2i:
                        encodingTemp.append(v2i[word])
                    else:
                        encodingTemp.append(v2i["<unk>"])
                    if len(encodingTemp) == maxLen - 1:
                        break
            # by the time you're out here, have looked at every word in this command
            commandTemp.append(encodingTemp)
            actionTemp.append(a2i[example[1][0]])
            targetTemp.append(t2i[example[1][1]])
        # once you're here, done looping through a set of commands, now to finalize the command/ action/ target temps
        # and then add them to the final things
        commandTemp.append(v2i["<end>"])
        actionTemp.append(v2i["<end>"])
        targetTemp.append(v2i["<end>"])

        # add them in
        trainEncoded.append(" ".join(commandTemp))
        trainOut.append([" ".join(actionTemp), " ".join(targetTemp)])


    valEncoded = list()
    valOut = list()
    for series in valUntoken:  # each set of commands/labels
        commandTemp = list()
        commandTemp.append(v2i["<start>"])
        actionTemp = list()
        actionTemp.append(v2i["<start>"])
        targetTemp = list()
        targetTemp.append(v2i["<start>"])
        for example in series:  # each pair of command and labels
            temp = preprocess_string(example[0])
            encodingTemp = list()
            wordList = temp.split()
            for word in wordList:
                word = word.lower()
                if len(word) > 0:
                    if word in v2i:
                        encodingTemp.append(v2i[word])
                    else:
                        encodingTemp.append(v2i["<unk>"])
                    if len(encodingTemp) == maxLen - 1:
                        break
            # by the time you're out here, have looked at every word in this command
            commandTemp.append(encodingTemp)
            actionTemp.append(a2i[example[1][0]])
            targetTemp.append(t2i[example[1][1]])
        # once you're here, done looping through a set of commands, now to finalize the command/ action/ target temps
        # and then add them to the final things
        commandTemp.append(v2i["<end>"])
        actionTemp.append(v2i["<end>"])
        targetTemp.append(v2i["<end>"])

        # add them in
        valEncoded.append(" ".join(commandTemp))
        valOut.append([" ".join(actionTemp), " ".join(targetTemp)])

    # converting the lists into np arrays
    trainEncoded = np.array(trainEncoded)
    trainOut = np.array(trainOut)
    valEncoded = np.array(valEncoded)
    valOut = np.array(valOut)

    trainDS = torch.utils.data.TensorDataset(torch.from_numpy(trainEncoded), torch.from_numpy(trainOut))
    valDS = torch.utils.data.TensorDataset(torch.from_numpy(valEncoded), torch.from_numpy(valOut))
    train_loader = torch.utils.data.DataLoader(trainDS, shuffle=True, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(valDS, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    model = md.EncoderDecoder()
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.CrossEntropyLoss()  # fixme is it okay to only have one?
    optimizer = None

    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter."
    # Output: "<BOS> GoToLocation diningtable <EOS>"
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        output = model(inputs, labels)

        loss = criterion(output.squeeze(), labels[:, 0].long())

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: write code to compute the accuracy of your model predictions
        # by comparing the predicted logits against the ground truth labels
        """
        acc = None

        # logging
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    epoch_loss /= len(loader)
    epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)