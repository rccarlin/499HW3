# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import numpy as np

# based off this: https://github.com/lkulowski/LSTM_encoder_decoder/blob/master/code/lstm_encoder_decoder.py
# because I have no clue how to code this and every place I've looked gave something different to decoder
# and I'm losing my mind

class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, inDim, hidDim):
        super(Encoder, self).__init__()
        # self.length = maxLen  # fixme is this really what you want
        # self.embedding = nn.Embedding(vocabSize, embedDim)
        self.inDim = inDim
        self.lstm = nn.LSTM(inDim, hidDim)
        # self.fcl = nn.Linear(hidDim, 1)  # right cos we only want one thing from encoder?

    def forward(self, x):
        # embeds = self.embedding(x)
        allOut, hid = self.lstm(x.view(x.shape[0], x.shape[1], self.inDim))
        # oneOut = self.fcl(allOut.view(len(x), -1))

        return hid  # fixme make sure we don't need to do a max?




class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, inDim, hidDim):
        super(Decoder, self).__init__()
        # self.embedding = nn.Embedding(vocabSize, embedDim)
        self.inDim = inDim
        self.lstm = nn.LSTM(inDim, hidDim)
        # self.fcl = nn.Linear(hidDim, inDim)  # fixme should it be vocab size?

    def forward(self, x, start):
        # embeds = self.embedding(x)
        allOut, hid = self.lstm(x.unsqueeze(0), start)  # fixme, not sure If I need the squeeze
        # oneOut = self.fcl(allOut.squeeze(0))

        return hid



class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, inDim, hidDim, numAct, numTar):
        super(EncoderDecoder, self).__init__()
        self.hidDim = hidDim
        self.encoder = Encoder(inDim, hidDim)
        self.decoder = Decoder(inDim, hidDim)
        self.actFCL = nn.Linear(hidDim, numAct)  # FIXME what dimensions
        self.tarFCL = nn.Linear(hidDim, numTar)

    def forward(self, ins, outs):  # is given one batch
        # what is encoder hidden?
        # decoderInput is ins[-1, :, :]
        # decoder his is encoder hidden

        # predicting actions:
        # she had for t in range numPredictions, but we just want to predict until end is predicted?
        # does that mean we gotta pass in the dictionaries?
        # oh I guess we can add condition where if the last thing you predicted is end, return what you got in
        numEp = len(ins)  # hopefully this will tell me the current batch size
        actions = np.zeros((numEp, self.numAct, self.numPred))
        targets = np.zeros((numEp, self.numTar, self.numPred))
        initialEnc = np.zeros((numEp, self.hidDim))  # I hope this is the correct shape, may supposed to do ep_len instead
        hidEnc = self.encoder(ins, initialEnc, outs)
        hidDec = hidEnc
        for p in range(self.numPred):
            actions[:, :, p] = self.actFCL(hidDec)
            targets[:, :, p] = self.tarFCL(hidDec)
            hidDec = self.decoder([outs[:,0,p], outs[:,1,p]], hidDec)

        return actions, targets
