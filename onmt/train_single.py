#!/usr/bin/env python
"""Training on a single process."""
import torch

from onmt.inputters.inputter import IterOnDevice
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from onmt.dynamic.fields import load_fields
from onmt.dynamic.iterator import build_dynamic_dataset_iter


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def _load_checkpoint(ckpt_path):
    """Load checkpoint if any."""
    checkpoint = None
    if ckpt_path:
        logger.info('Loading checkpoint from %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def _get_model_opts(opt, checkpoint=None):
    """Get `model_opt` to build model, may load from `checkpoint` if any."""
    if checkpoint is not None:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
    else:
        model_opt = opt
    return model_opt


def _build_valid_iter(opt, fields, device_id):
    """Build iterator used for validation."""
    valid_iter = build_dynamic_dataset_iter(
        fields, opt, is_train=False)
    return valid_iter


def _build_train_iter(opt, fields, stride=1, offset=0):
    """Build training iterator."""
    train_iter = build_dynamic_dataset_iter(
        fields, opt, is_train=True, stride=stride, offset=offset)
    return train_iter


def get_train_iter(opt, stride=1, offset=0):
    """Return training iterator."""
    checkpoint = _load_checkpoint(ckpt_path=opt.train_from)
    fields = load_fields(opt.save_data, checkpoint)
    train_iter = _build_train_iter(opt, fields, stride=stride, offset=offset)
    return train_iter


def main(opt, device_id, batch_queue=None, semaphore=None):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    checkpoint = _load_checkpoint(ckpt_path=opt.train_from)
    fields = load_fields(opt.save_data, checkpoint)

    model_opt = _get_model_opts(opt, checkpoint=checkpoint)
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
    model.count_parameters(log=logger.info)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver)

    if batch_queue is None:
        _train_iter = _build_train_iter(opt, fields)
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

    valid_iter = _build_valid_iter(opt, fields, device_id)
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
