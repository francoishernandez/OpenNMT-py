#!/usr/bin/env python
"""Train models using alternate dynamic data loader."""
import os
import signal
import torch

from pprint import pformat

import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.train_single import main as single_main
from onmt.utils.parse import ArgumentParser

from onmt.dynamicdata.iterators import build_mixer
from onmt.dynamicdata.dataset import build_data_loader, \
    build_dataset_adaptor_iter

from itertools import cycle

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def train(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        #logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        #fields = checkpoint['vocab']
        if 'data_loader_step' in checkpoint['opt']:
            data_loader_step = checkpoint['opt'].data_loader_step
            logger.info('Set data_loader_step {} from model_opt'.format(
                data_loader_step))
        elif opt.data_loader_step is not None:
            data_loader_step = opt.data_loader_step
            logger.info('Overrode data_loader_step {} from opt'.format(
                data_loader_step))
        else:
            data_loader_step = checkpoint['optim']['training_step']
            logger.info('Approximated data_loader_step {} from optim. '
                        'Consider using --data_loader_step'.format(
                            data_loader_step))
    else:
        data_loader_step = 0

    nb_gpu = len(opt.gpu_ranks)

    # always using producer/consumer
    queues = []
    mp = torch.multiprocessing.get_context('spawn')
    semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    # Train with multiprocessing.
    procs = []
    for device_id in range(nb_gpu):
        q = mp.Queue(opt.queue_size)
        queues += [q]
        procs.append(mp.Process(target=run, args=(
            opt, device_id, error_queue, q, semaphore,), daemon=True))
        procs[device_id].start()
        logger.info(" Starting process pid: %d  " % procs[device_id].pid)
        error_handler.add_child(procs[device_id].pid)
    producer = mp.Process(target=batch_producer,
                          args=(queues, semaphore, opt, data_loader_step),
                          daemon=True)
    producer.start()
    error_handler.add_child(producer.pid)

    for p in procs:
        p.join()
    producer.terminate()


def batch_producer(queues, semaphore, opt, data_loader_step):
    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)

    data_config, transforms, dataset_adaptor = build_data_loader(opt)
    logger.info('Transforms:')
    logger.info(pformat(transforms))
    mixer, task_epochs = build_mixer(data_config, transforms, is_train=True, bucket_size=opt.bucket_size)
    report_every = max(opt.queue_size, opt.report_every)

    def mb_callback(i):
        if i % report_every == 0:
            logger.info('mb %s epochs %s',
                  i, {key: ge.epoch for key, ge in task_epochs.items()})
            for task in transforms:
                logger.info('* transform stats ({})'.format(task))
                for transform in transforms[task]:
                    for line in transform.stats():
                        logger.info('\t{}'.format(line))

    train_iter = build_dataset_adaptor_iter(
        mixer, dataset_adaptor, opt, mb_callback, data_loader_step, is_train=True)

    def next_batch():
        new_batch = next(train_iter)
        semaphore.acquire()
        return new_batch

    b = next_batch()

    for device_id, q in cycle(enumerate(queues)):
        b.dataset = None
        # the training process has the gpu, and is responsible for calling torch.device

        # hack to dodge unpicklable `dict_keys`
        b.fields = list(b.fields)
        q.put(b)
        b = next_batch()


def run(opt, device_id, error_queue, batch_queue, semaphore):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id, batch_queue, semaphore, dynamic=True)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    train(opt)


if __name__ == "__main__":
    main()
