import torch
from typing import Optional
import lm_eval
from lm_eval.models.huggingface import HFLM as eval_wrapper
from lm_eval.tasks import get_task_dict, TaskManager
from lm_eval.evaluator import evaluate
import os, json
from data.lm_loader import DataLoaderArgs, LMLoader
from data.tokenizer import Tokenizer
from config import parse_eval_args
from arch.model import ModelArgs, Model, create_kv_cache
import glob
import re
import wandb
import math
import tqdm

def device_mapping(cuda_device: int):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """

    def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:  # pylint: disable=unused-argument
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage

    return inner_device_mapping

class EvalWrapper(eval_wrapper):
    def __init__(
        self,
        model,
        tokenizer,
        batch_size,
        max_seq_length: Optional[int]=None,
    ):
        super().__init__(pretrained="gpt2")
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device('cuda')
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length
        self._batch_size = batch_size
        self._rank = 0
        self._world_size = 1

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id
    
    @property
    def eos_token_id(self):
        return self._tokenizer.eos_id
    
    @property
    def pad_token_id(self):
        return self._tokenizer.pad_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 1024

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device
    
    @property
    def model(self):
        return self._model

    def tok_encode(self, string: str, **kwargs):
        encoded = self._tokenizer.encode(string, bos=True, eos=False)
        return encoded
    
    def tok_decode(self, tokens, **kwargs):
        if type(tokens) == int:
            tokens = [tokens]
        decoded = self._tokenizer.decode(tokens)
        return decoded
    
    def tok_batch_encode(self, strings, left_truncate_len=None, **kwargs):
        tokens = [self._tokenizer.encode(string, bos=True, eos=False) for string in strings]
        if left_truncate_len is not None:
            tokens = [t[-left_truncate_len:] for t in tokens]
        max_len = max(len(t) for t in tokens)
        tokens = [t + [self.pad_token_id] * (max_len - len(t)) for t in tokens]
        tensor = torch.tensor(tokens).long()
        return tensor, tensor # return a dummy tensor for the attention mask

    def _model_call(self, inps):
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        with ctx:
            logits = self.model(inps)
        return logits

    def _model_generate(self, context, max_length, **generation_kwargs):
        bsz = context.size(0)
        tokens = context.tolist()
        # remove padding
        tokens = [t[:t.index(self.pad_token_id)] if self.pad_token_id in t else t for t in tokens]
        min_prompt_len = min([len(t) for t in tokens])
        max_prompt_len = max([len(t) for t in tokens])

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        kv_cache = create_kv_cache(self.model.args, bsz)
        output = torch.zeros(bsz, max_length, dtype=torch.long, device=self.device).fill_(self.pad_token_id)
        output[:, :max_prompt_len] = context[:, :max_prompt_len]
        input_text_mask = output != self.pad_token_id
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

        for cur_pos in range(min_prompt_len, max_length):
            with ctx:
                logits = self.model(output[:, prev_pos:cur_pos], start_pos=prev_pos, kv_cache=kv_cache)
            next_tokens = logits[:, -1, :].argmax(dim=-1).reshape(-1)
            next_tokens = torch.where(input_text_mask[:, cur_pos], output[:, cur_pos], next_tokens)
            output[:, cur_pos] = next_tokens
            eos_reached |= (next_tokens == self.eos_token_id) & (~input_text_mask[:, cur_pos])
            prev_pos = cur_pos
            if eos_reached.all():
                break

        return output


def _adjust_config(task_dict):
    adjusted_task_dict = {}
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            adjusted_task_dict = {
                **adjusted_task_dict,
                **{task_name: _adjust_config(task_obj)},
            }
        else:
            if 'mmlu' in task_name or 'gsm' in task_name:
                task_obj.set_config(key="num_fewshot", value=5)
            elif task_obj.get_config("num_fewshot") is None:
                task_obj.set_config(key="num_fewshot", value=0)
            task_obj.set_fewshot_seed(seed=1234)
            adjusted_task_dict[task_name] = task_obj

    return adjusted_task_dict

@torch.no_grad()
def eval_end_task(
    model,
    tokenizer,
    tasks,
    limit,
    batch_size,
    max_seq_length,
):
    model_eval_wrapper = EvalWrapper(
        model,
        tokenizer,
        batch_size,
        max_seq_length,
    )

    task_dict = get_task_dict(tasks.split(','), task_manager=TaskManager(verbosity='WARNING'))
    task_dict = _adjust_config(task_dict)

    eval_results = evaluate(
        model_eval_wrapper,
        task_dict,
        limit=limit,
        verbosity='WARNING',
    )
    return eval_results

@torch.no_grad()
def eval_ppl(
    model,
    dataloader,
    limit,
):
    total_loss, total_tokens = 0, 0
    for i, batch in tqdm.tqdm(enumerate(dataloader)):
        if limit is not None and i >= limit:
            break
        tokens, targets, loss_mask = batch
        loss, n_tokens = model(tokens, targets=targets, loss_mask=loss_mask)
        total_loss += loss.item()
        total_tokens += n_tokens
    ppl = math.exp(total_loss / total_tokens)
    return ppl

def load_model(checkpoint_dir: str, merging_checkpoint_dir: Optional[str]=None):
    with open(os.path.join(checkpoint_dir, "metadata.json")) as f:
        metadata = json.load(f)

    if any('moe' in key for key in metadata["modelargs"].keys()):
        modelargs = MoEModelArgs(**metadata["modelargs"])
    else:
        modelargs = ModelArgs(**metadata["modelargs"])
    model = Model(modelargs)
    model = model.cuda()
    model.eval()
    model_path = os.path.join(checkpoint_dir, "model_state.pt")
    model_state = torch.load(model_path, map_location=device_mapping(-1))
    if merging_checkpoint_dir is not None:
        merging_checkpoint_dir = [os.path.join(ckpt_dir, "model_state.pt") for ckpt_dir in merging_checkpoint_dir.split(',')]
        for ckpt in merging_checkpoint_dir:
            _state = torch.load(ckpt, map_location=device_mapping(-1))
            for key in model_state.keys():
                model_state[key] += _state[key]
        for key in model_state.keys():
            model_state[key] /= (len(merging_checkpoint_dir) + 1)
    model.load_state_dict(model_state)

    dataloader_args = DataLoaderArgs(**metadata["dataloader_args"])
    tokenizer = Tokenizer(dataloader_args.tokenizer_path)
    updates = metadata["updates"]
    return model, tokenizer, updates

def build_valid_loader(args, checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "metadata.json")) as f:
        metadata = json.load(f)

    dataloader_args = DataLoaderArgs(**metadata["dataloader_args"])
    dataloader_args.data_path = args.valid_set
    dataloader_args.batch_size = args.batch_size

    loader = LMLoader(dataloader_args, prefetch=False, shuffle=False, device='cuda', num_shards=1, shard_id=0)
    return loader

def evaluate_one_checkpoint(args):
    model, tokenizer, updates = load_model(args.checkpoint_dir, args.merging_checkpoint_dir)

    if args.tasks is not None:
        results = eval_end_task(
            model,
            tokenizer,
            args.tasks,
            args.limit,
            args.batch_size,
            model.args.max_seq_len,
        )
        
        for task, res in results["results"].items():
            if task in args.tasks.split(','):
                print(f"{task}: {res}")

        return results, updates
    elif args.valid_set is not None:
        valid_loader = build_valid_loader(args, args.checkpoint_dir)
        valid_ppl = eval_ppl(
            model,
            valid_loader,
            args.limit,
        )
        print(f"valid_ppl: {valid_ppl}")
        return None, updates

def version_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

if __name__ == '__main__':
    args = parse_eval_args()
    
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, id=args.wandb_id, name=args.wandb_id, reinit=True)

    if os.path.exists(os.path.join(args.checkpoint_dir, "metadata.json")):
        results, updates = evaluate_one_checkpoint(args)
    else:
        checkpoint_folders = glob.glob(os.path.join(args.checkpoint_dir, "updates_*"))
        checkpoint_folders = sorted(checkpoint_folders, key=version_key)
        for checkpoint_folder in checkpoint_folders:
            args.checkpoint_dir = checkpoint_folder
            results, updates = evaluate_one_checkpoint(args)

            for task, res in results["results"].items():
                if task in args.tasks.split(','):
                    acc = res['acc_norm,none'] if 'acc_norm,none' in res else res['acc,none']
                    if args.wandb_project is not None:
                        wandb.log({f"valid/{task}": acc}, step=updates)

    if args.wandb_project is not None:
        wandb.finish()
