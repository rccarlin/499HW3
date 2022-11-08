# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, embedDim, vocabSize, maxLen, numBatch):
        super(Encoder, self).__init__()
        self.embedding_dim = embedDim
        self.vocab_size = vocabSize
        self.length = maxLen  # fixme is this really what you want
        self.num_batches = numBatch
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.GRU = nn.GRU(self.length)

    def forward(self, x):
        pass


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, embedDim, vocabSize, maxLen, numBatch):
        super(Decoder, self).__init__()
        self.embedding_dim = embedDim
        self.vocab_size = vocabSize
        self.length = maxLen
        self.num_batches = numBatch

    def forward(self, x):
        pass


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, embedDim, vocabSize, maxLen, numBatch):
        super(EncoderDecoder, self).__init__()
        self.embedding_dim = embedDim
        self.vocab_size = vocabSize
        self.length = maxLen
        self.num_batches = numBatch
        self.encoder = Encoder(self.embedding_dim, self.vocab_size, self.length, self.num_batches)
        self.decoder = Decoder(self.embedding_dim, self.vocab_size, self.length, self.num_batches)


    def forward(self, x):
        pass
