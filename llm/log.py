import time
import logging
import math
import sys
from collections import OrderedDict
from numbers import Number
from torch.utils.tensorboard import SummaryWriter
import os
import atexit
import wandb

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

_tensorboard_writers = {}
def _close_writers():
    for w in _tensorboard_writers.values():
        w.close()

atexit.register(_close_writers)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class Logger:

    def __init__(self, args, rank=0):
        self.args = args
        self.rank = rank
        self.tensorboard_dir = os.path.join(args.checkpoint_dir, 'tsb') if rank == 0 else None
        if self.tensorboard_dir is not None:
            os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.tensorboard_writer = _tensorboard_writers

        if rank == 0 and args.wandb_project is not None:
            run_name = os.environ.get('WANDB_NAME', os.path.basename(args.checkpoint_dir))
            self.wandb_run = wandb.init(project=args.wandb_project, reinit=False, name=run_name, entity=args.wandb_entity, settings=wandb.Settings(init_timeout=120))
        else:
            self.wandb_run = None

        log_level = logging.INFO if rank == 0 else logging.WARNING
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self.meters = OrderedDict()
        self.valid_meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_ppl'] = AverageMeter()
        self.meters['train_moe_loss'] = AverageMeter()
        self.meters['throughput'] = TimeMeter()       # words per second
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['lr'] = 0
        self.meters['wd'] = 0
        self.meters['gate_loss_weight'] = 0
        self.valid_meters['valid_loss'] = AverageMeter()
    
    def _format_stats(self, stats):
        def format_stat(stat):
            if isinstance(stat, Number):
                stat = '{:g}'.format(stat)
            elif isinstance(stat, AverageMeter):
                stat = '{:.3f}'.format(stat.avg)
            elif isinstance(stat, TimeMeter):
                stat = '{:g}'.format(round(stat.avg))
            return stat

        postfix = OrderedDict(stats)
        for key in postfix.keys():
            postfix[key] = str(format_stat(postfix[key]))
        return postfix
    
    def _get_tensorboard_writer(self, tag):
        if self.tensorboard_dir is None:
            return None
        if tag not in self.tensorboard_writer:
            self.tensorboard_writer[tag] = SummaryWriter(os.path.join(self.tensorboard_dir, tag))
        return self.tensorboard_writer[tag]

    def log_metrics(self, stats, tag='', step=0):
        def _str_pipes(stats):
            return ' | '.join(key + ' ' + stats[key].strip()
                          for key in stats.keys())

        # Only rank 0 outputs to console
        if self.rank == 0:
            postfix = _str_pipes(self._format_stats(stats))
            self._logger.info('{} | step {} | {}'.format(tag, step, postfix))

        # Log to tensorboard
        writer = self._get_tensorboard_writer(tag)
        if writer is None:
            return
        for key in stats.keys():
            if isinstance(stats[key], Number):
                value = stats[key]
            elif isinstance(stats[key], AverageMeter):
                value = stats[key].avg
            elif isinstance(stats[key], TimeMeter):
                value = round(stats[key].avg)
            else:
                raise NotImplementedError
            writer.add_scalar(key, value, step)
        writer.flush()

        # Log to wandb
        if self.wandb_run is not None:
            prefix = "" if tag is None else tag + "/"
            for key in stats.keys():
                if isinstance(stats[key], AverageMeter):
                    self.wandb_run.log({prefix + key: stats[key].avg}, step=step)
                elif isinstance(stats[key], TimeMeter):
                    self.wandb_run.log({prefix + key: round(stats[key].avg)}, step=step)
                elif isinstance(stats[key], Number):
                    self.wandb_run.log({prefix + key: stats[key]}, step=step)

    def log_text(self, text):
        self._logger.info(text)

    def update_meters(self, logging_dicts):
        self.meters['train_loss'].update(logging_dicts['loss'] / logging_dicts['tokens'] / math.log(2), 
                                         logging_dicts['tokens'])

        avg_loss_log2 = logging_dicts['loss'] / logging_dicts['tokens'] / math.log(2)      
        ppl = 2 ** avg_loss_log2
        self.meters['train_ppl'].update(ppl, logging_dicts['tokens'])

        self.meters['train_moe_loss'].update(logging_dicts['moe_loss'] / logging_dicts['tokens'], 
                                         logging_dicts['tokens'])
        self.meters['throughput'].update(logging_dicts['tokens'])
        self.meters['bsz'].update(logging_dicts['tokens'])
        self.meters['gnorm'].update(logging_dicts['grad_norm'])
        self.meters['lr'] = logging_dicts['lr']
        self.meters['wd'] = logging_dicts['wd']
        self.meters['gate_loss_weight'] = logging_dicts['gate_loss_weight']

    def update_valid_meters(self, logging_dicts):
        self.valid_meters['valid_loss'].update(logging_dicts['loss'] / logging_dicts['tokens'] / math.log(2), 
                                               logging_dicts['tokens'])

    def reset_meters(self, meters):
        for m in meters:
            if hasattr(meters[m], 'reset'):
                meters[m].reset()
