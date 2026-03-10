import copy
import json
import itertools
import os
from dataclasses import dataclass
import random
import math
import torch

import numpy as np
from .infinibatch import (
    CheckpointableIterator, 
    PrefetchIterator, 
    MapIterator, 
    BlockwiseShuffleIterator, 
    FixedBatchIterator,
    SelectManyIterator,
    WeightIterator,
    NativeCheckpointableIterator,
    MultiplexIterator,
    ChunkedSourceIterator,
    BufferedShuffleIterator
)
from .tokenizer import Tokenizer

@dataclass
class DataLoaderArgs:
    data_path: str = None
    data_weight: str = "1.0"
    tokenizer_path: str = None
    max_seq_len: int = 2048
    batch_read_ahead: int = 1000
    seed: int = 1
    batch_size: int = 8
    num_epochs: int = 100

class LMLoader(CheckpointableIterator):
    def __init__(
        self,
        args,
        seed=1,
        shuffle=True,
        prefetch=True,
        num_shards=1,
        shard_id=0,
        device='cuda'
    ):
        super().__init__()
        self.args = args
        self.data = self._prepare_data(args.data_path, args.data_weight)
        self.tokenizer = Tokenizer(args.tokenizer_path)

        self.seed = str(seed)
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.device = device

        self._build_iter()

    def _prepare_data(self, data_path, data_weight):
        data_path = [json.load(open(x, "r")) for x in data_path.split(',')]
        data_weight = [float(x) for x in data_weight.split(',')]
        assert len(data_path) == len(data_weight), "data_path and data_weight should have the same length"
        data = []
        for i in range(len(data_path)):
            for d in data_path[i]:
                d['weight'] *= data_weight[i]
                data.append(d)
        return data

    def _build_iter(self):
        tokenized_lines = self._tokenize_and_sampling()
        padded_batches = self._batchify(tokenized_lines)

        if self.prefetch:
            prefetch_batches = PrefetchIterator(
                padded_batches,
                buffer_size=10000,
                buffer_in_main_process=True,
                log_empty_buffer_warning=True and self.shard_id == 0,
            )
        else:
            prefetch_batches = padded_batches

        prefetch_batches = MapIterator(prefetch_batches, self._move_to_tensor)

        self._iter = prefetch_batches

    def _tokenize_and_sampling(self):
        multiple_iters = []
        weights = []

        for data in self.data:
            multiple_iters.append(self._tokenize(data))
            weights.append(float(data["weight"]))

        if len(multiple_iters) == 1:
            return multiple_iters[0]

        sampling_iterator = WeightIterator(weights, self.seed)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        tokenized_lines = MultiplexIterator(
            control_iterator, multiple_iters
        )

        return tokenized_lines

    def _tokenize(self, data):
        dataset = data["source"]

        if self.shuffle:
            _random = random.Random(self.seed)
            dataset = dataset * math.ceil(self.args.num_epochs)
            _random.shuffle(dataset)

        chunk_files = ChunkedSourceIterator(
            dataset,
            num_instances=self.num_shards,
            instance_rank=self.shard_id,
        )

        tokenized_lines = SelectManyIterator(
            chunk_files, lambda files: self._read_from_files(files)
        )

        return tokenized_lines

    def _batchify(self, lines):

        if self.shuffle:
            lines = BlockwiseShuffleIterator(
                lines, self.args.batch_read_ahead, self.seed
            )

        batches = FixedBatchIterator(lines, self.args.batch_size)

        padded_batches = MapIterator(batches, self.collate)

        return padded_batches

    def collate(self, batch):
        batch_size = len(batch)

        input_ids = np.full(
            shape=(batch_size, self.args.max_seq_len),
            dtype=np.int32,
            fill_value=self.tokenizer.pad_id,
        )

        target_ids = np.full(
            shape=(batch_size, self.args.max_seq_len),
            dtype=np.int32,
            fill_value=self.tokenizer.pad_id,
        )

        loss_mask = np.full(
            shape=(batch_size, self.args.max_seq_len),
            dtype=np.bool,
            fill_value=False,
        )

        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)-1] = ids[:-1]
            target_ids[i, : len(ids)-1] = ids[1:]
            loss_mask[i, : len(ids)-1] = True
        
        input_ids = input_ids.astype(np.int64)
        target_ids = target_ids.astype(np.int64)

        return input_ids, target_ids, loss_mask

    def _read_from_files(self, file_path):

        if not os.path.exists(file_path):
            print("| file {} not exists".format(file_path), flush=True)
            return iter([])  # skip bad file

        data = []

        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                if 'text' not in line:
                    text = line['content']
                else:
                    text = line['text']
                tokens = self.tokenizer.encode(text, bos=True, eos=True)
                data += tokens
                while len(data) > self.args.max_seq_len:
                    yield data[:self.args.max_seq_len+1]
                    data = data[self.args.max_seq_len:]

        if len(data) > 1:
            yield data

    def _move_to_tensor(self, batch):
        return tuple(torch.tensor(x, device=self.device) for x in batch)

    @property
    def iterator(self):
        if self._iter is None:
            raise NotImplementedError("_build_iter() must called first")
        return self._iter

    def __iter__(self):
        if self._iter is None:
            raise NotImplementedError("_build_iter() must called first")
        return self._iter

    def __next__(self):
        return next(self._iter)

    def setstate(self, value):
        self._iter.setstate(value)

    def getstate(self):
        return self._iter.getstate()

    def close(self):
        self._iter.close()
