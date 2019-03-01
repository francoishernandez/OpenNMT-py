"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        # HERE WE NEED TO ADD SOME CNN LAYERS FOR AUDIO
        self.cnn1 = nn.Conv2d(1, 1, 3, stride=2)
        self.cnn2 = nn.Conv2d(1, 1, 3, stride=2)
        # self.cnn3 = nn.Conv2d(1, 1, 2, stride=2)
        # self.audio_embeddings = nn.Linear(39, d_model)
        self.audio_embeddings = nn.Linear(39, d_model)

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        
        # print("src before conv", src.size())
        
        src = src.unsqueeze(1)
        # print("src unsqueeze", src.size())
        src = src.permute(2, 1, 3, 0)
        # print("src permute", src.size())

        conv1 = self.cnn1(src)
        # print("conv1", conv1.size())
        relu1 = nn.functional.relu(conv1)
        # print("relu1", relu1.size())
        conv2 = self.cnn2(relu1)
        # print("conv2", conv2.size())
        flat_conv2 = conv2.reshape(conv2.size(3), conv2.size(0), -1)
        # print("flat_conv2", flat_conv2.size())
        emb = self.audio_embeddings(flat_conv2)

        words = flat_conv2[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        # padding_idx = self.embeddings.word_padding_idx
        padding_idx = 1
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]

        # print("src", src.size())
        # emb = self.audio_embeddings(src)
        # print("emb", emb.size())
        out = emb.transpose(0, 1).contiguous()
        # print("out", out.size())

        # print("mask", mask.size())
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        # print("out decoder", out.size())

        return emb, out.transpose(0, 1).contiguous(), lengths, flat_conv2
