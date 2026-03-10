import os
import math
import json
import argparse
import torch
from biteval.eval_utils import get_test_dataset
import random

from typing import Optional
from arch.model import ModelArgs, Model
from data.lm_loader import DataLoaderArgs
from data.tokenizer import Tokenizer


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


def load_model(checkpoint_dir: str, merging_checkpoint_dir: Optional[str]=None, tokenizer_path: Optional[str]=None):
    with open(os.path.join(checkpoint_dir, "metadata.json")) as f:
        metadata = json.load(f)

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
    
    # udpated_model_state = {}
    # for key, value in model_state.items():
    #     udpated_model_state[key.replace("model.", "")] = value

    model.load_state_dict(model_state)

    dataloader_args = DataLoaderArgs(**metadata["dataloader_args"])
    if tokenizer_path is not None:
        dataloader_args.tokenizer_path = tokenizer_path
    tokenizer = Tokenizer(dataloader_args.tokenizer_path)
    updates = metadata["updates"]
    return model, tokenizer, updates

from tqdm import tqdm
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='xxx/bitnet_b1_58', type=str)
parser.add_argument('--seqlen', default=2048, type=int)
parser.add_argument('--output_sparsity', default=False, action='store_true')
parser.add_argument('--sparse_ratio', default=-1, type=float)
parser.add_argument('--gate_sparse_ratio', default=-1, type=float)
parser.add_argument('--input_bits', default=-1, type=int)
parser.add_argument('--dataset', default="wikitext2,c4", type=str)
parser.add_argument('--bf16', default=False, action='store_true')
parser.add_argument('--checkpoint_dir', type=str, required=True,
                    help='Path to the model checkpoint directory')
parser.add_argument('--merging_checkpoint_dir', type=str, default=None)
parser.add_argument('--tokenizer_path', type=str, default=None,
                    help='Path to the tokenizer model file. If None, will use the tokenizer from the checkpoint metadata.')


def calulate_loss(model, input, target, loss_fct, output_sparsity=False):
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = model(
                input,
                # use_cache=False,
                # output_hidden_states=False,
                # output_attentions=False, 
                # output_sparsity=output_sparsity
            )
    # output = output[0]
    shift_logits = output
    shift_labels = target
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    num_tokens = (shift_labels > 0).sum().item()
    loss = loss.float() * (shift_labels > 0).float().view(-1)
    loss = loss.sum()
    return loss, num_tokens


def main(args):
    if args.dataset != "":
        datasets = args.dataset.split(",")
    else:
        datasets = ['wikitext2', 'c4']
    
    model, tokenizer, updates = load_model(args.checkpoint_dir, args.merging_checkpoint_dir, args.tokenizer_path)
    model = model.cuda()

    print(f"tokenizer bos id = {tokenizer.bos_id}, eos id = {tokenizer.eos_id}, pad id = {tokenizer.pad_id}, unk id = {tokenizer.unk_id}")

    # tokenizer = tokenizer.tok
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none").cuda()

    ppl = []
    for dataset in datasets:
        testdata, all_target_labels = get_test_dataset(dataset, tokenizer, seqlen=args.seqlen)
        acc_loss, count = 0.0, 0
        batch_size = 32
        progress = tqdm(range(0, len(testdata), batch_size))
        # all_sparsity = [{"query": 0, "key": 0, "value": 0, "outprj": 0, "gate": 0, "up": 0, "down": 0, "gate_output": 0, "query_output": 0, "key_output": 0, "value_output": 0, "outprj_output": 0, "up_output": 0, "down_output": 0,} for _ in range(model.config.num_hidden_layers)]
        for ii in progress:
            input_tensors = []
            target_tensors = []
            max_len = 0
            for jj in range(0, batch_size):
                if ii + jj >= len(testdata):
                    break
                input_tensors.append(torch.Tensor(testdata[ii + jj]).long().cuda().view(1, -1))
                target_tensors.append(torch.Tensor(all_target_labels[ii + jj]).long().cuda().view(1, -1))
                max_len = max(max_len, input_tensors[-1].size(-1))
            if len(input_tensors) == 0:
                continue
            for kk in range(len(input_tensors)):
                if input_tensors[kk].size(-1) < max_len:
                    input_tensors[kk] = torch.cat([input_tensors[kk], torch.zeros(1, max_len - input_tensors[kk].size(-1)).long().cuda()], dim=-1)
                    target_tensors[kk] = torch.cat([target_tensors[kk], torch.zeros(1, max_len - target_tensors[kk].size(-1)).long().cuda()], dim=-1)
            input = torch.cat(input_tensors, dim=0)
            target = torch.cat(target_tensors, dim=0)

            loss, num_tokens = calulate_loss(model, input, target, loss_fct, output_sparsity=args.output_sparsity)
            # if args.output_sparsity:
            #     for layer_idx in range(model.config.num_hidden_layers):
            #         for key in list(all_sparsity[layer_idx].keys()):
            #             if running_sparsity[layer_idx][key] is not None:
            #                 all_sparsity[layer_idx][key] += running_sparsity[layer_idx][key]
            count += num_tokens
            acc_loss += loss.item()
            avg_loss = acc_loss / count / math.log(2)          
            progress.set_description(f"avg_loss = {acc_loss/ count / math.log(2)}")

        # if args.output_sparsity:
        #     for layer_idx in range(model.config.num_hidden_layers):
        #         for key in list(all_sparsity[layer_idx].keys()):
        #             all_sparsity[layer_idx][key]  = round(all_sparsity[layer_idx][key] / len(testdata), 2)
        #         print(layer_idx, all_sparsity[layer_idx])
        #     total_sparsity = {"query": 0, "key": 0, "value": 0, "outprj": 0, "gate": 0, "up": 0, "down": 0, "gate_output": 0, "query_output": 0, "key_output": 0, "value_output": 0, "outprj_output": 0, "up_output": 0, "down_output": 0,}
        #     for layer_idx in range(model.config.num_hidden_layers):
        #         for key in list(all_sparsity[layer_idx].keys()):
        #             total_sparsity[key] += all_sparsity[layer_idx][key]
        #     for key in list(total_sparsity.keys()):
        #         total_sparsity[key] = round(total_sparsity[key] / model.config.num_hidden_layers, 2)
        #     print("{} Sparsity: {}".format(dataset, total_sparsity))
            # for key, value in total_sparsity.items():
            #     total_sparsity[key] = 1 - value / 100
            # total_compute = (total_sparsity["query"] + total_sparsity["key"] + total_sparsity["value"] + total_sparsity["outprj"] ) * model.config.hidden_size * model.config.hidden_size 
            # if model.config.glu:
            #     total_compute += (total_sparsity["down"] + total_sparsity["gate"] + total_sparsity["up"]) * model.config.intermediate_size * model.config.hidden_size
            #     total_compute /= (3 * model.config.hidden_size * model.config.intermediate_size + 4 * model.config.hidden_size * model.config.hidden_size)
            # else:
            #     total_compute += (total_sparsity["down"] + total_sparsity["up"]) * model.config.intermediate_size * model.config.hidden_size
            #     total_compute /= (2 * model.config.hidden_size * model.config.intermediate_size + 4 * model.config.hidden_size * model.config.hidden_size)
            # print("Total:", round(total_compute * 100, 2))
        avg_loss = acc_loss / count / math.log(2)
        ppl.append(2 ** avg_loss)
        print("{} PPL: {}".format(dataset, ppl[-1]))

    print(ppl)
    print("Avg PPL:", sum(ppl) / len(ppl))


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)