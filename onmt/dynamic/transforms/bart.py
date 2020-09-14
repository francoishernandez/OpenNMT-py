"""Transforms relate to noising from BART: based on code of fairseq."""
import math
import numpy as np
import torch
from functools import partial
from onmt.dynamic.transforms import register_transform
from .transform import Transform


MASK_TOK = '<mask>'
SUBWORD_SPACER = '▁'
SUBWORD_JOINER = '￭'


def word_start(x, ignore_subword=False, is_joiner=False):
    """Return if a token is the word start or not."""
    if not ignore_subword:
        if is_joiner:
            return not x.startswith(SUBWORD_JOINER)
        else:
            return x.startswith(SUBWORD_SPACER)
    else:
        return True


class BARTNoising(object):
    """Noise from BART."""

    def __init__(self, vocab, mask_tok=MASK_TOK, mask_ratio=0.0,
                 insert_ratio=0.0, permute_sent_ratio=0.0, poisson_lambda=3.0,
                 replace_length=-1, rotate_ratio=0.5, mask_length='subword',
                 random_ratio=0.0, is_joiner=False,
                 full_stop_token=[".", "?", "!"]):
        self.vocab = vocab

        self.mask_tok = mask_tok

        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.insert_ratio = insert_ratio
        self.rotate_ratio = rotate_ratio
        self.permute_sent_ratio = permute_sent_ratio

        self.full_stop_token = full_stop_token

        # -1: keep everything (i.e. 1 mask per token)
        #  0: replace everything (i.e. no mask)
        #  1: 1 mask per span
        if replace_length not in [-1, 0, 1]:
            raise ValueError(f'invalid arg: replace_length={replace_length}')
        self.replace_length = replace_length

        if mask_length not in ['subword', 'word', 'span-poisson']:
            raise ValueError(f'invalid arg: mask-length={mask_length}')
        if mask_length == 'subword' and replace_length not in [0, 1]:
            raise ValueError(f'if using subwords, use replace-length=1 or 0')

        if mask_length == 'subword' or is_joiner is None:
            # view each subword as word start / input is word level token
            self.__is_word_start = partial(word_start, ignore_subword=True)
        else:
            self.__is_word_start = partial(word_start, is_joiner=is_joiner)

        self.mask_span_distribution = None
        if mask_length == 'span-poisson':
            self.mask_span_distribution = self._make_poisson(poisson_lambda)

    def _make_poisson(self, poisson_lambda):
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-poisson_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= poisson_lambda
            k_factorial *= (k + 1)
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        return torch.distributions.Categorical(ps)

    def _is_full_stop(self, token):
        return True if token in self.full_stop_token else False

    def permute_sentences(self, tokens, p=1.0):
        if len(tokens) == 1:
            return tokens
        full_stops = np.array([self._is_full_stop(token) for token in tokens])
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-1] = True

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero()[0] + 2

        n_sentences = sentence_ends.size
        if n_sentences == 1:
            return tokens

        n_to_permute = math.ceil((n_sentences * 2 * p) / 2.0)

        substitutions = np.random.permutation(n_sentences)[:n_to_permute]
        ordering = np.arange(0, n_sentences)
        ordering[substitutions] = substitutions[np.random.permutation(
            n_to_permute)]

        result = [tok for tok in tokens]
        index = 0
        for i in ordering:
            sentence = tokens[(sentence_ends[i - 1] if i > 0 else 0):
                              sentence_ends[i]]
            result[index:index + len(sentence)] = sentence
            index += len(sentence)
        assert len(result) == len(tokens), "Error when permute sentences."
        return result

    def _is_word_start(self, token):
        return self.__is_word_start(token)

    def whole_word_mask(self, tokens, p=1.0):  # text span mask/infilling
        is_word_start = torch.tensor(
            [self._is_word_start(token) for token in tokens]).int()
        n_mask = int(math.ceil(is_word_start.sum() * p))
        n_insert = 0
        if n_mask == 0:
            return tokens

        if self.mask_span_distribution is not None:  # Text (span) Infilling
            lengths = self.mask_span_distribution.sample(
                sample_shape=(n_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < n_mask:
                lengths = torch.cat([
                    lengths,
                    self.mask_span_distribution.sample(
                        sample_shape=(n_mask,))
                ], dim=0)
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < n_mask:
                i += 1
            lengths[i] = n_mask - (0 if i == 0 else cum_length[i - 1])
            n_mask = i + 1
            lengths = lengths[:n_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            n_insert = n_mask - lengths.size(0)
            n_mask -= n_insert
            if n_mask == 0:
                return self.insertion_noise(tokens, n_insert / len(tokens))

            assert (lengths > 0).all()
        else:  # Token Masking
            lengths = torch.ones((n_mask,)).long()
        # assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[torch.randperm(word_starts.size(0))[
            :n_mask]].squeeze(1)
        mask_random = torch.FloatTensor(n_mask).uniform_() < self.random_ratio

        tokens_length = len(tokens)
        # assert tokens_length - 1 not in indices
        to_keep = torch.ones(tokens_length, dtype=torch.bool)

        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            for i in indices:
                tokens[i] = self.mask_tok
            random_toks = torch.randint(
                0, len(self.vocab), size=(mask_random.sum(),))
            for i, rand_tok in zip(indices[mask_random], random_toks):
                tokens[i] = rand_tok

        if tokens_length - 1 in indices:
            uncompleted = (indices != tokens_length - 1)
            indices = indices[uncompleted]
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]

        # acts as a long length, so spans don't go over the end of doc
        is_word_start[-1] = 255

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1  # 1 for the position already masked
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                # next position from each word_start
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token: 1 mask/remove per span
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]: 1 mask per token
                    for i in indices:
                        tokens[i] = self.mask_tok
                    random_toks = torch.randint(
                        0, len(self.vocab), size=(mask_random.sum(),))
                    for i, rand_tok in zip(indices[mask_random], random_toks):
                        tokens[i] = rand_tok
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                # to cover whole token
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    for i in indices:
                        tokens[i] = self.mask_tok
                    random_toks = torch.randint(
                        0, len(self.vocab), size=(mask_random.sum(),))
                    for i, rand_tok in zip(indices[mask_random], random_toks):
                        tokens[i] = rand_tok

                # assert tokens_length - 1 not in indices

        tokens = [tok for tok, keep in zip(tokens, to_keep)
                  if keep.item() is True]

        if n_insert > 0:
            tokens = self.insertion_noise(tokens, n_insert / len(tokens))

        return tokens

    def insertion_noise(self, tokens, p=1.0):
        if p == 0.0:
            return tokens

        n_tokens = len(tokens)
        n_insert = int(math.ceil(n_tokens * p))
        n_random = int(math.ceil(n_insert * self.random_ratio))

        noise_indices = np.random.permutation(n_tokens + n_insert)[:n_insert]
        noise_mask = np.zeros(shape=(n_tokens + n_insert,), dtype=bool)
        noise_mask[noise_indices] = 1

        result = np.empty(shape=(n_tokens + n_insert,), dtype=object)
        result[noise_indices[n_random:]] = self.mask_tok
        if n_random > 0:
            result[noise_indices[:n_random]] = np.random.choice(
                self.vocab, size=n_random)
        result[~noise_mask] = tokens

        assert all([item is not None for item in result]),\
            "Error when insert noise."
        return [tok for tok in result]

    def rolling_noise(self, tokens, p=1.0):
        if np.random.random() >= p:
            return tokens
        offset = np.random.randint(0, max(1, len(tokens) - 1) + 1)
        return tokens[offset:] + tokens[0:offset]

    def apply(self, tokens):
        if self.vocab is None:
            raise ValueError("Inject BART noise require a valid vocabulary.")

        if self.permute_sent_ratio > 0.0:
            tokens = self.permute_sentences(tokens, self.permute_sent_ratio)

        if self.mask_ratio > 0.0:
            tokens = self.whole_word_mask(tokens, self.mask_ratio)

        if self.insert_ratio > 0.0:
            tokens = self.insertion_noise(tokens, self.insert_ratio)

        if self.rotate_ratio > 0.0:
            tokens = self.rolling_noise(tokens, self.rotate_ratio)
        return tokens


@register_transform(name='bart')
class BARTNoiseTransform(Transform):
    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to BART."""
        group = parser.add_argument_group('Transform/BART')
        group.add('--permute_sent_ratio', '-permute_sent_ratio',
                  type=float, default=0.0,
                  help="shuffle this proportion of sentences in all inputs")
        group.add('--rotate_ratio', '-rotate_ratio', type=float, default=0.5,
                  help="rotate this proportion of inputs")
        group.add('--insert_ratio', '-insert_ratio', type=float, default=0.0,
                  help="insert this percentage of additional random tokens")
        group.add('--random_ratio', '-random_ratio', type=float, default=0.0,
                  help="instead of using [MASK], use random token this often")

        group.add('--mask_ratio', '-mask_ratio', type=float, default=0.0,
                  help="fraction of words/subwords that will be masked")
        group.add('--mask_length', '-mask_length', type=str, default='subword',
                  choices=['subword', 'word', 'span-poisson'],
                  help="mask length to choose")
        group.add('--poisson_lambda', '-poisson_lambda',
                  type=float, default=0.0,
                  help="randomly shuffle sentences for this proportion.")
        group.add('--replace_length', '-replace_length',
                  type=int, default=-1, choices=[-1, 0, 1],
                  help="when masking N tokens, replace with 0, 1, or N tokens."
                  "(use -1 for N)")

    def warm_up(self, vocabs):
        self.vocab = vocabs

        subword_type = self.opts.src_subword_type
        if self.opts.mask_length == 'subword':
            if subword_type == 'none':
                raise ValueError(
                    f'src_subword_type={subword_type} incompatible with '
                    f'mask_length={self.opts.mask_length}!')
        is_joiner = (subword_type == 'bpe') if subword_type != 'none' else None
        self.bart_noise = BARTNoising(
            vocabs,
            mask_tok=MASK_TOK,
            mask_ratio=self.opts.mask_ratio,
            insert_ratio=self.opts.insert_ratio,
            permute_sent_ratio=self.opts.permute_sent_ratio,
            poisson_lambda=self.opts.poisson_lambda,
            replace_length=self.opts.replace_length,
            rotate_ratio=self.opts.rotate_ratio,
            mask_length=self.opts.mask_length,
            random_ratio=self.opts.random_ratio,
            is_joiner=is_joiner
        )

    def apply(self, src, tgt, is_train=False, stats=None, **kwargs):
        """Apply switchout to both src and tgt side tokens."""
        if is_train and self.vocab is not None:
            src = self.bart_noise.apply(src)
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return ''
