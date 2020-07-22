"""Module for dynamic data transfrom."""
import os
import torch
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
            src (list): a list of tokens;
            tgt (list): a list of tokens;
        """
        raise NotImplementedError

    def stats(self):
        """Return statistic message."""
        return None


class SimpleTransform(Transform):
    """A naive transform class for debug use."""

    @classmethod
    def get_specials(cls, opts):
        return ({'<SPECIAL1>', '<SPECIAL2>'}, set())

    def warm_up(self, vocabs):
        print("warm up simple transform.")

    def apply(self, src, tgt, **kwargs):
        return src, tgt


class SwitchOutTransform(Transform):
    def __init__(self, opts):
        super().__init__(opts)

    def warm_up(self, vocabs):
        self.vocab = vocabs

    def _swithout(self, tokens):
        return tokens

    def apply(self, src, tgt, **kwargs):
        src = self._swithout(src)
        tgt = self._swithout(tgt)
        return src, tgt

    def stats(self):
        pass


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


AVAILABLE_TRANSFORMS = {
    'simple': SimpleTransform,
    'switchout': SwitchOutTransform,
    'sentencepiece': SentencePieceTransform
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
    return all_specials


def save_transforms(opts, transforms):
    """Dump `transforms` object."""
    transforms_path = "{}.transforms.pt".format(opts.save_data)
    os.makedirs(os.path.dirname(transforms_path), exist_ok=True)
    torch.save(transforms, transforms_path)


def load_transforms(opts):
    """Load dumped `transforms` object."""
    transforms_path = "{}.transforms.pt".format(opts.save_data)
    transforms = torch.load(transforms_path)
    # Logger for transforms
    return transforms
