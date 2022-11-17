# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import numpy as np
import torch

# based off this: https://github.com/lkulowski/LSTM_encoder_decoder/blob/master/code/lstm_encoder_decoder.py
# because I have no clue how to code this and every place I've looked gave something different to decoder
# and I'm losing my mind

class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, hidDim, embedDim, vocabSize):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embedDim)
        self.lstm = nn.LSTM(embedDim, hidDim, batch_first=True)

    def forward(self, x, start):  # originally had , start, goal
        embeds = self.embedding(x)
        # allOut, hid = self.lstm(x.view(x.shape[0], x.shape[1], self.inDim))
        allOut, hid = self.lstm(embeds)
        return hid  # fixme make sure we don't need to do a max?




class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, hidDim, embedDim, vocabSize):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embedDim)
        self.lstm = nn.LSTM(embedDim, hidDim, batch_first=True)
        self.vocabSize = vocabSize
        self.embedDim = embedDim

    def forward(self, x, start):
        # print(self.vocabSize, self.embedDim)
        embeds = self.embedding(x)
        allOut, hid = self.lstm(embeds, start)  # fixme, not sure If I need the squeeze

        return allOut, hid



class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, hidDim, numAct, numTar, numPred, embedDim, numWords):
        super(EncoderDecoder, self).__init__()
        self.hidDim = hidDim
        self.encoder = Encoder(hidDim, embedDim, numWords)
        self.decoder = Decoder(hidDim, embedDim, numAct + numTar)
        self.actFCL = nn.Linear(94, numAct)  # FIXME what dimensions
        self.tarFCL = nn.Linear(94, numTar)
        self.flatten = nn.Linear(hidDim, 1)
        self.numPred = numPred
        self.numAct = numAct
        self.numTar = numTar
        self.embedDim = embedDim

    def forward(self, ins, outs, teacherForcing=True):  # is given one batch

        numEp = len(ins)  # hopefully this will tell me the current batch size
        actions = torch.zeros((numEp, self.numAct, self.numPred))
        targets = torch.zeros((numEp, self.numTar, self.numPred))
        initialEnc = (np.zeros((numEp, self.hidDim)), np.zeros((numEp, self.hidDim)))
        hidEnc = self.encoder(ins, initialEnc)
        #torch.set_printoptions(profile="full")

        # hidEnc = self.encoder(ins)[0]  # fixme is this the one I want?
        hidDec = hidEnc  # fixme is this what I want I can't even tell
        inDec = torch.zeros(((numEp,94))).int()
        for p in range(self.numPred):
            out, hidDec = self.decoder(inDec, hidDec)
            out = self.flatten(out).squeeze()
            tempact = self.actFCL(out)
            tempTar = self.tarFCL(out)
            actions[:, :, p] = tempact
            targets[:, :, p] = tempTar

            # make one hot for teacher forcing
            oneHotOut = np.zeros((numEp,94))  # 94 is numAct + numTar... I hope
            i = 0
            for o in outs[:,:,p]:
                oneHotAct = [0 for a in range(self.numAct)]
                oneHotTar = [0 for t in range(self.numTar)]
                oneHotAct[o[0]] = 1
                oneHotTar[o[1]] = 1
                oneHotOut[i] = oneHotAct + oneHotTar
                i += 1
            inDec = torch.from_numpy(oneHotOut).int()


        return actions, targets
