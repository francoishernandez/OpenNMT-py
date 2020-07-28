"""Module for dynamic data transfrom."""
import os
import numpy as np
import torch
from onmt.utils.logging import logger
from onmt.dynamic.vocab import get_vocabs


class Transform(object):
    """A Base class that every transform method should derived from."""
    def __init__(self, opts):
        self.opts = opts

    def warm_up(self, vocabs):
        pass

    @classmethod
    def get_specials(cls, opts):
        return (set(), set())

    def apply(self, src, tgt, **kwargs):
        """Apply transform to `src` & `tgt`.

        Args:
            src (list): a list of str, representing tokens;
            tgt (list): a list of str, representing tokens;
        """
        raise NotImplementedError

    def stats(self):
        """Return statistic message."""
        return None

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return ''

    def __repr__(self):
        cls_name = type(self).__name__
        cls_args = self._repr_args()
        return '{}({})'.format(cls_name, cls_args)


class HammingDistanceSampling(object):
    """Functions related to (negative) Hamming Distance Sampling."""

    def _softmax(self, x):
        softmax = np.exp(x)/sum(np.exp(x))
        return softmax

    def _sample_replace(self, vocab, reject):
        """Sample a token from `vocab` other than `reject`."""
        token = reject
        while token == reject:
            token = np.random.choice(vocab)
        return token

    def _sample_distance(self, tokens, temperature):
        """Sample number of tokens to corrupt from `tokens`."""
        n_tokens = len(tokens)
        indices = np.arange(n_tokens)
        logits = indices * -1 * temperature
        probs = self._softmax(logits)
        distance = np.random.choice(indices, p=probs)
        return distance

    def _sample_position(self, tokens, distance):
        n_tokens = len(tokens)
        indices = np.arange(n_tokens)
        chosen_indices = np.random.choice(
            indices, size=distance, replace=False)
        return chosen_indices


class SwitchOutTransform(Transform, HammingDistanceSampling):
    """SwitchOut."""

    def __init__(self, opts):
        super().__init__(opts)

    def warm_up(self, vocabs):
        self.vocab = vocabs
        self.temperature = self.opts.switchout_temperature

    def _switchout(self, tokens, vocab):
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. sample corrupted values
        out = []
        for (i, tok) in enumerate(tokens):
            if i in chosen_indices:
                tok = self._sample_replace(vocab, reject=tok)
                out.append(tok)
            else:
                out.append(tok)
        return out

    def apply(self, src, tgt, **kwargs):
        """Apply switchout to both src and tgt side tokens."""
        # import pdb; pdb.set_trace()
        src = self._switchout(src, self.vocab['src'])
        tgt = self._switchout(tgt, self.vocab['tgt'])
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('switchout_temperature', self.temperature)

    def stats(self):
        pass


class TokenDropTransform(Transform, HammingDistanceSampling):
    """Random drop tokens from sentence."""

    def __init__(self, opts):
        super().__init__(opts)
        self.temperature = self.opts.tokendrop_temperature

    def _token_drop(self, tokens):
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. Drop token on chosen position
        out = [tok for (i, tok) in enumerate(tokens)
               if i not in chosen_indices]
        return out

    def apply(self, src, tgt, **kwargs):
        """Apply token drop to both src and tgt side tokens."""
        src = self._token_drop(src)
        tgt = self._token_drop(tgt)
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('worddrop_temperature', self.temperature)

    def stats(self):
        pass


class TokenMaskTransform(Transform, HammingDistanceSampling):
    """Random mask tokens from src sentence."""

    MASK_TOK = '｟MASK｠'

    def __init__(self, opts):
        super().__init__(opts)
        self.temperature = opts.tokenmask_temperature

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by prefix transform."""
        return ({cls.MASK_TOK}, set())

    def _token_mask(self, tokens):
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. mask word on chosen position
        out = []
        for (i, tok) in enumerate(tokens):
            tok = self.MASK_TOK if i in chosen_indices else tok
            out.append(tok)
        return out

    def apply(self, src, tgt, **kwargs):
        """Apply word drop to both src and tgt side tokens."""
        src = self._token_mask(src)
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('tokenmask_temperature', self.temperature)


class PrefixSrcTransform(Transform):
    """Add target language Prefix to src sentence."""

    def __init__(self, opts):
        super().__init__(opts)
        self.prefix_dict = self.get_prefix_dict(self.opts)

    @classmethod
    def _get_prefix(cls, corpus):
        """Get prefix string of a `corpus`."""
        prefix_tag_format = "｟_tgt_is_{}｠"
        if 'prefix' in corpus['transforms']:
            tl = corpus['tgt_lang']
            prefix = prefix_tag_format.format(tl)
        else:
            prefix = None
        return prefix

    @classmethod
    def get_prefix_dict(cls, opts):
        """Get all needed prefix correspond to corpus in `opts`."""
        prefix_dict = {}
        for c_name, corpus in opts.data.items():
            prefix = cls._get_prefix(corpus)
            if prefix is not None:
                logger.info(f"Get prefix for {c_name}: {prefix}")
                prefix_dict[c_name] = prefix
        return prefix_dict

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by prefix transform."""
        prefix_dict = cls.get_prefix_dict(opts)
        src_specials = set(prefix_dict.values())
        return (src_specials, set())

    def _prepend(self, tokens, corpus_name):
        """Get prefix according to `corpus_name` and prepend to `tokens`."""
        corpus_prefix = self.prefix_dict.get(corpus_name, None)
        if corpus_prefix is None:
            raise ValueError('corpus_name does not exist.')
        return [corpus_prefix] + tokens

    def apply(self, src, tgt, **kwargs):
        """Prepend prefix to src side tokens."""
        corpus_name = kwargs.get('corpus_name', None)
        if corpus_name is None:
            raise ValueError('corpus_name is required.')
        src = self._prepend(src, corpus_name)
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('prefix_dict', self.prefix_dict)


class SentencePieceTransform(Transform):
    """A sentencepiece subword transform class."""

    def __init__(self, opts):
        """Initialize neccessary options for sentencepiece."""
        super().__init__(opts)
        self.share_vocab = opts.share_vocab
        self.src_subword_model = opts.src_subword_model
        self.tgt_subword_model = opts.tgt_subword_model
        self.n_samples = opts.n_samples_subword
        self.theta = opts.theta_subword

    def warm_up(self, vocabs=None):
        """Load subword models."""
        import sentencepiece as spm
        load_src_model = spm.SentencePieceProcessor()
        load_src_model.Load(self.src_subword_model)
        if self.share_vocab:
            self.load_models = {
                'src': load_src_model,
                'tgt': load_src_model
            }
        else:
            load_tgt_model = spm.SentencePieceProcessor()
            load_tgt_model.Load(self.tgt_subword_model)
            self.load_models = {
                'src': load_src_model,
                'tgt': load_tgt_model
            }

    def _sentencepiece(self, tokens, side='src'):
        """Do sentencepiece subword encode."""
        sp_model = self.load_models[side]
        sentence = ' '.join(tokens)
        segmented = sp_model.SampleEncodeAsPieces(
            sentence, nbest_size=self.n_samples, alpha=self.theta)
        return segmented

    def apply(self, src, tgt, **kwargs):
        """Apply sentencepiece subword encode to src & tgt."""
        src = self._sentencepiece(src, 'src')
        tgt = self._sentencepiece(tgt, 'tgt')
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}, {}={}, {}={}, {}={}'.format(
            'share_vocab', self.share_vocab,
            'n_samples', self.n_samples,
            'theta', self.theta,
            'src_subword_model', self.src_subword_model,
            'tgt_subword_model', self.tgt_subword_model
        )

    def __getstate__(self):
        """Pickling following for rebuild."""
        return {'opts': self.opts,
                'share_vocab': self.share_vocab,
                'src_subword_model': self.src_subword_model,
                'tgt_subword_model': self.tgt_subword_model,
                'n_samples': self.n_samples,
                'theta': self.theta}

    def __setstate__(self, d):
        """Reload when unpickling from save file."""
        self.opts = d['opts']
        self.share_vocab = d['share_vocab']
        self.src_subword_model = d['src_subword_model']
        self.tgt_subword_model = d['tgt_subword_model']
        self.n_samples = d['n_samples']
        self.theta = d['theta']
        self.warm_up()


class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)
        self.src_seq_length = opts.src_seq_length
        self.tgt_seq_length = opts.tgt_seq_length

    def apply(self, src, tgt, **kwargs):
        """Return None if too long else return as is."""
        if len(src) > self.src_seq_length or len(tgt) > self.tgt_seq_length:
            return None
        else:
            return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format(
            'src_seq_length', self.src_seq_length,
            'tgt_seq_length', self.tgt_seq_length
        )


AVAILABLE_TRANSFORMS = {
    'sentencepiece': SentencePieceTransform,
    'switchout': SwitchOutTransform,
    'tokendrop': TokenDropTransform,
    'tokenmask': TokenMaskTransform,
    'prefix': PrefixSrcTransform,
    'filtertoolong': FilterTooLongTransform
}


def make_transforms(opts, transforms_cls, fields):
    """Build transforms in `transforms_cls` with vocab of `fields`."""
    vocabs = get_vocabs(fields)
    transforms = {}
    for name, transform_cls in transforms_cls.items():
        transform_obj = transform_cls(opts)
        transform_obj.warm_up(vocabs)
        transforms[name] = transform_obj
    return transforms


def get_transforms_cls(transform_names):
    """Return valid transform class indicated in `transform_names`."""
    transforms_cls = {}
    for name in transform_names:
        if name not in AVAILABLE_TRANSFORMS:
            raise ValueError("specified tranform not supported!")
        transforms_cls[name] = AVAILABLE_TRANSFORMS[name]
    return transforms_cls


def get_specials(opts, transforms_cls_dict):
    """Get specials of transforms that should be registed in Vocab."""
    all_specials = {'src': set(), 'tgt': set()}
    for name, transform_cls in transforms_cls_dict.items():
        src_specials, tgt_specials = transform_cls.get_specials(opts)
        all_specials['src'].update(src_specials)
        all_specials['tgt'].update(tgt_specials)
    logger.info(f"Get special vocabs from Transforms: {all_specials}.")
    return all_specials


def save_transforms(opts, transforms):
    """Dump `transforms` object."""
    transforms_path = "{}.transforms.pt".format(opts.save_data)
    os.makedirs(os.path.dirname(transforms_path), exist_ok=True)
    logger.info(f"Saving Transforms to {transforms_path}.")
    torch.save(transforms, transforms_path)


def load_transforms(opts):
    """Load dumped `transforms` object."""
    transforms_path = "{}.transforms.pt".format(opts.save_data)
    transforms = torch.load(transforms_path)
    logger.info("Transforms loaded.")
    return transforms
