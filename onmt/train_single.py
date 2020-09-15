#!/usr/bin/env python
"""Training on a single process."""
import os

import torch

from onmt.inputters.inputter import IterOnDevice
from onmt.inputters.iterator import build_dataset_iter,\
    build_dataset_iter_multiple
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from onmt.dynamic.vocab import load_fields
from onmt.dynamic.iterator import build_dynamic_dataset_iter


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def _load_checkpoint(opt):
    """Load checkpoint if any."""
    checkpoint = None
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def _load_fields(opt, checkpoint, dynamic=False):
    """Load fields from preprocess file/checkpoint."""
    if dynamic:
        # should already verified data_config
        fields = load_fields(opt)
    else:
        if checkpoint is not None:
            logger.info(f'Loading vocab from checkpoint at {opt.train_from}.')
            vocab = checkpoint['vocab']
        else:
            vocab = torch.load(opt.data + '.vocab.pt')
        fields = vocab
    return fields


def _get_model_opts(opt, checkpoint=None):
    """Get `model_opt` to build model, may load from `checkpoint` if any."""
    if checkpoint is not None:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
    else:
        model_opt = opt
    return model_opt


def _build_valid_iter(opt, fields, device_id, dynamic=False):
    """Build iterator used for validation."""
    if dynamic:
        valid_iter = build_dynamic_dataset_iter(
            fields, opt, is_train=False)
    else:
        valid_iter = build_dataset_iter(
            "valid", fields, opt, is_train=False)
    return valid_iter


def _build_train_iter(opt, fields, dynamic=False, stride=1, offset=0):
    """Build training iterator."""
    if dynamic:
        train_iter = build_dynamic_dataset_iter(
            fields, opt, is_train=True, stride=stride, offset=offset)
    else:
        if len(opt.data_ids) > 1:
            train_shards = []
            for train_id in opt.data_ids:
                shard_base = "train_" + train_id
                train_shards.append(shard_base)
            train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
        else:
            if opt.data_ids[0] is not None:
                shard_base = "train_" + opt.data_ids[0]
            else:
                shard_base = "train"
            train_iter = build_dataset_iter(shard_base, fields, opt)
    return train_iter


def get_train_iter(opt, dynamic=False, stride=1, offset=0):
    """Return training iterator."""
    checkpoint = _load_checkpoint(opt)
    fields = _load_fields(opt, checkpoint, dynamic=dynamic)
    train_iter = _build_train_iter(
        opt, fields, dynamic=dynamic, stride=stride, offset=offset)
    return train_iter


def main(opt, device_id, batch_queue=None, semaphore=None, dynamic=False):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    checkpoint = _load_checkpoint(opt)
    model_opt = _get_model_opts(opt, checkpoint=checkpoint)

    fields = _load_fields(opt, checkpoint, dynamic=dynamic)
    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver)

    if batch_queue is None:
        _train_iter = _build_train_iter(opt, fields, dynamic=dynamic)
        train_iter = IterOnDevice(_train_iter, device_id)
    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                # Move batch to specified device
                IterOnDevice.batch_to_device(batch, device_id)
                yield batch

        train_iter = _train_iter()

    valid_iter = _build_valid_iter(opt, fields, device_id, dynamic=dynamic)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, device_id)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
