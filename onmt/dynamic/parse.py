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
            path_src = corpus.get('path_src', None)
            path_tgt = corpus.get('path_tgt', None)
            if path_src is None or path_tgt is None:
                raise ValueError(f'Corpus {cname} path should be specified.')
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
