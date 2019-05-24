#!/usr/bin/env python
"""Train models."""
import os
import signal
import torch

import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.logging import logger
from onmt.train_single import main as single_main
from onmt.utils.parse import ArgumentParser

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab

# import multiprocessing
# from multiprocessing import Pipe, Process
import torch.multiprocessing as mp

REQUEST_GET_NEXT = '__request_next__'
REQUEST_BREAK = '__request_break__'
RESPONSE_DATA = '__response_data__'
RESPONSE_DONE = '__response_done__'

def generator_proxy(pipe_to_generator_server):
    """
    Acts like a generator, but sends next() requests through the pipe to the
    generator server.
    """
    while True:
        pipe_to_generator_server.send((REQUEST_GET_NEXT,))
        response = pipe_to_generator_server.recv()
        if response[0] == RESPONSE_DATA:
            yield response[1]
        elif response[0] == RESPONSE_DONE:
            raise StopIteration
        else:
            raise ValueError('Invalid message cmd')

def proxy_break(pipe_to_generator_server):
    """
    Informs the generator server that the given pipe will no longer be querying
    the generator.  It's "break"ing out of it's local loop.
    """
    pipe_to_generator_server.send((REQUEST_BREAK,))

def generator_server(generator_to_serve, pipes_to_proxy_processes):
    """
    Recieves requests from generator_proxy instances and calls the generator
    on their behalf, sending the next() responses back to the proxys.
    """
    generator_to_serve = iter(generator_to_serve)
    running_pipes = list(pipes_to_proxy_processes)
    while len(running_pipes) > 0:
        pipes_to_remove = []
        for device_id, pipe in enumerate(running_pipes):
            if pipe.poll():
                msg = pipe.recv()
                if msg[0] == REQUEST_BREAK:
                    pipes_to_remove.append(pipe)
                elif msg[0] == REQUEST_GET_NEXT:
                    try:
                        device = torch.device(1)
                        generator_to_serve._next_device = device
                        val = next(generator_to_serve)
                        # print("STUFF WE WANT TO SEND")
                        # print(type(val))
                        # print(val)
                        # picklable_val = val
                        picklable_val = [val.src, val.tgt, val.indices]
                        # print("SRC", type(val.src))
                        # print("TGT", type(val.tgt))
                        # print("Indices", type(val.indices))
                        # for pv in picklable_val:
                        #     if type(pv) is tuple:
                        #         for item in pv:
                        #             item.to(torch.device(device_id))
                        #     else:
                        #         pv.to(torch.device(device_id))
                        
                        
                        # print(picklable_val)
                        pipe.send((RESPONSE_DATA, picklable_val))
                    except StopIteration:
                        pipes_to_remove.append(pipe)
                        pipe.send((RESPONSE_DONE,))
                else:
                    raise ValueError('Invalid message cmd')
        for p in pipes_to_remove:
            running_pipes.remove(p)

def main(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    nb_gpu = len(opt.gpu_ranks)

    # IDEA:
    # Spawn a generator process, that will be called by each process to retrieve their batch

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab
    train_iter = build_dataset_iter("train", fields, opt)

    mp.set_start_method('spawn', force=True)
    pipes = [mp.Pipe() for x in range(opt.world_size)]
    server_pipes = [p[1] for p in pipes]

    batch_server = mp.Process(target=generator_server,
        args=(train_iter, server_pipes))


    batch_server.start()
    print("BATCH SERVER PROCID", batch_server.pid)

    if opt.world_size > 1:
        # mp = torch.multiprocessing.get_context('spawn')
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, pipes[device_id][0]), daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        for p in procs:
            print("SOME GPU PROCID", p.pid, 0)
            p.join()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0, pipes[0][0])
    else:   # case only CPU
        single_main(opt, -1)


def run(opt, device_id, error_queue, server_pipe):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        print(gpu_rank)
        print(opt.gpu_ranks[device_id])
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        single_main(opt, device_id, server_pipe)
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


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
