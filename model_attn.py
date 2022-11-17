# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import numpy as np
import torch

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
        embeds = self.embedding(x)
        allOut, hid = self.lstm(embeds, start)
        alignment = torch.softmax(torch.matmul(allOut, start.t()), dim=-1)
        context = torch.matmul(alignment, start)
        # fixme I did a dot product attention method since I couldn't figure out how to do the attention with weights
        #  and technically the instructions did not tell me which to use
        return hid
    # and here inlines the the issue: I did-- or tried to-- all this math for attention but I don't actually use it
    #



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
        self.actFCL = nn.Linear(hidDim, numAct)
        self.tarFCL = nn.Linear(hidDim, numTar)
        self.numPred = numPred
        self.numAct = numAct
        self.numTar = numTar
        self.embedDim = embedDim

    def forward(self, ins, outs):  # is given one batch

        numEp = len(ins)  # hopefully this will tell me the current batch size
        actions = torch.zeros((numEp, self.numAct, self.numPred))
        targets = torch.zeros((numEp, self.numTar, self.numPred))
        initialEnc = (np.zeros((numEp, self.hidDim)), np.zeros((numEp, self.hidDim)))
        hidEnc = self.encoder(ins, initialEnc)

        hidDec = hidEnc
        for p in range(self.numPred):
            tempact = self.actFCL(hidDec[0])
            tempTar = self.tarFCL(hidDec[0])
            actions[:, :, p] = tempact
            targets[:, :, p] = tempTar

            # make one hot
            oneHotOut = np.zeros((numEp,94))  # 94 is numAct + numTar... I hope
            i = 0
            for o in outs[:,:,p]:
                oneHotAct = [0 for a in range(self.numAct)]
                oneHotTar = [0 for t in range(self.numTar)]
                oneHotAct[o[0]] = 1
                oneHotTar[o[1]] = 1
                oneHotOut[i] = oneHotAct + oneHotTar
                i += 1
            hidDec = self.decoder(torch.from_numpy(oneHotOut).int(), hidDec)

        return actions, targets
