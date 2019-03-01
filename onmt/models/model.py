""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        # print("##### FORWARD MODEL")
        # print("src", src.size())
        # print("tgt", tgt.size())
        if str(type(self.encoder)) == "<class 'onmt.encoders.audio_encoder.AudioEncoder'>":
            enc_state, memory_bank, lengths = self.encoder(src, lengths)
            src_conv = src
        else:
            src = src.permute(3, 0, 1, 2)
            src = src.reshape(src.size(0), src.size(1), -1)
            enc_state, memory_bank, lengths, src_conv = self.encoder(src, lengths)
        # print("permuted reshaped src", src.size())
        # print("enc_state", enc_state.size())
        # print("memory_bank", memory_bank.size())
        # print("lengths", lengths)
        if bptt is False:
            self.decoder.init_state(src_conv, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns
