"""Parse cls for dynamic."""
from onmt.utils.parse import ArgumentParser
from onmt.utils.logging import logger


class DynamicArgumentParser(ArgumentParser):

    @classmethod
    def valid_dynamic_corpus(cls, opt):
        """Parse corpus specified in data field of YAML file."""
        import yaml
        corpora = yaml.safe_load(opt.data)
        for cname, corpus in corpora.items():
            # Check path
            path_src = corpus.get('path_src', None)
            path_tgt = corpus.get('path_tgt', None)
            if path_src is None or path_tgt is None:
                raise ValueError(f'Corpus {cname} path are required')
            # Check language
            src_lang = corpus.get('src_lang', None)
            tgt_lang = corpus.get('tgt_lang', None)
            if src_lang is None or tgt_lang is None:
                raise ValueError(f'Corpus {cname} lang info are required.')
            # Check weight
            weight = corpus.get('weight', None)
            if weight is None:
                logger.warning(f"Corpus {cname}'s weight should be given."
                               " We default it to 1 for you.")
                corpus['weight'] = 1
        logger.info(f"Parsed {len(corpora)} corpora from -data.")
        opt.data = corpora

    @classmethod
    def get_all_transform(self, opt):
        """Should only called after `valid_dynamic_corpus`."""
        global_transform = opt.transforms
        if len(global_transform) != 0:
            logger.info(f"Global transforms: {global_transform}.")
        all_transforms = set()
        for cname, corpus in opt.data.items():
            _transforms = set(corpus.get('transforms', []))
            if len(_transforms) == 0 and len(global_transform) != 0:
                corpus['transforms'] = global_transform
            all_transforms.update(_transforms)
        all_transforms.update(global_transform)
        opt._all_transform = all_transforms
