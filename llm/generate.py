from arch.model import ModelArgs, Model, create_kv_cache
from data.tokenizer import Tokenizer
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer model file')
parser.add_argument('--model_path', type=str, required=True, help='Path to model weights (.pth file)')
args = parser.parse_args()

tokenizer = Tokenizer(args.tokenizer_path)
model = Model(ModelArgs())
model.cuda()
model.load_state_dict(torch.load(args.model_path))

# print(tokenizer.decode(tokenizer.encode('Hello, world!', bos=True, eos=True)))
ids = tokenizer.encode('What is the', bos=True, eos=False)
output_ids = [id for id in ids]
ids = torch.tensor(ids).cuda().unsqueeze(0)
print(ids)

kv_cache = create_kv_cache(model.args, 1)
start_pos = 0

for i in range(100):
    output = model(ids, start_pos=start_pos, kv_cache=kv_cache)
    start_pos += ids.size(1)
    ids = output[:,-1,:].argmax(-1).unsqueeze(1)
    output_ids.append(ids.squeeze().item())

print(tokenizer.decode(output_ids))

