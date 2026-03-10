import torch
import transformers

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm 
from datasets import load_dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_test_dataset(dataset_name, tokenizer, seqlen=2048):
    if dataset_name == "wikitext2":
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testdata = "".join(testdata['text']).split('\n')
    elif dataset_name == "c4":
        testdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')['text']
    else:
        raise NotImplementedError
    
    testdata = [item for item in testdata if item != ""]
    tokenized_text = [tokenizer.encode(item)['input_ids'] for item in testdata]
    # sort by length
    tokenized_text = sorted(tokenized_text, key=lambda x: len(x), reverse=True)
    print(f"EOS token id: {tokenizer.eos_token_id}")
    print(f"BOS token id: {tokenizer.bos_token_id}")
    all_input_tokens = []
    all_target_labels = []

    for doc_tokens in tokenized_text:
        input_tokens = [tokenizer.bos_token_id] + doc_tokens
        target_labels = doc_tokens + [tokenizer.eos_token_id]
        if len(input_tokens) > seqlen:
            input_tokens = input_tokens[:seqlen]
            target_labels = target_labels[:seqlen]
        all_input_tokens.append(input_tokens)
        all_target_labels.append(target_labels)

        if len(all_input_tokens) < 2:
            print(f"Input tokens: {input_tokens}"
                  f"\nTarget labels: {target_labels}\n")

    return all_input_tokens, all_target_labels


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from tokenization_bitnet import BitnetTokenizer 
    tokenizer = BitnetTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from tokenization_bitnet import BitnetTokenizer 
    tokenizer = BitnetTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from tokenization_bitnet import BitnetTokenizer
    tokenizer = BitnetTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from tokenization_bitnet import BitnetTokenizer
    tokenizer = BitnetTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from tokenization_bitnet import BitnetTokenizer
    tokenizer = BitnetTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)


def get_test_tokens(
    name, seed=0, seqlen=2048, model='',
):
    train_samples = 0
    if name == 'wikitext2':
        return get_wikitext2(train_samples, seed, seqlen, model)[1].input_ids
    elif name == 'ptb':
        return get_ptb_new(train_samples, seed, seqlen, model)[1].input_ids
    elif name == 'c4':
        return get_c4_new(train_samples, seed, seqlen, model)[1].input_ids
    else:
        raise Exception

try:
    from lm_eval import utils
    from lm_eval.base import BaseLM
    class LMEvalAdaptor(BaseLM):
        def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
            super().__init__()

            assert isinstance(batch_size, int)

            self.model_name = model_name
            self.model = model
            self.model.eval()

            self.tokenizer = tokenizer

            self.vocab_size = self.tokenizer.vocab_size

            self._batch_size = batch_size

            self._max_length = max_length

        @property
        def eot_token_id(self):
            # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            if self._max_length != -1:
                return self._max_length
            if hasattr(self.model.config, "n_ctx"):
                return self.model.config.n_ctx
            elif hasattr(self.model.config, "max_position_embeddings"):
                return self.model.config.max_position_embeddings
            elif hasattr(self.model.config, "n_positions"):
                return self.model.config.n_positions
            elif "bloom" in self.model_name:
                return 2048
            elif "llama" in self.model_name:
                return 2048  # TODO: did not check this
            elif "mpt" in self.model_name:
                return 2048
            elif "falcon" in self.model_name:
                return 2048
            else:
                print(self.model.config)
                raise NotImplementedError

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def device(self):
            return "cuda"

        def tok_encode(self, string: str, add_special_tokens=True):
            return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens)

        def loglikelihood(self, requests):
            new_reqs = []
            for context, continuation in requests:
                context = context.strip()
                if context == "":
                    # end of text as context
                    context_enc = [self.eot_token_id]
                else:
                    context_enc = self.tok_encode(context, add_special_tokens=True)

                continuation = continuation.strip()
                continuation_enc = self.tok_encode(continuation, add_special_tokens=False)

                new_reqs.append(((context, continuation), context_enc, continuation_enc))

            return self._loglikelihood_tokens(new_reqs)

        def _model_call(self, inps):
            """
            inps: a torch tensor of shape [batch, sequence]
            the size of sequence may vary from call to call

            returns: a torch tensor of shape [batch, sequence, vocab] with the
            logits returned from the model
            """
            with torch.no_grad():
                out = self.model(inps)[0]
                # out = self.model(inps)[0][0]
            return out

        def _model_generate(self, context, max_length, eos_token_id):
            return self.model.generate(
                context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
            )
        
        def greedy_until(self, requests):
            # TODO: implement fully general `until` that handles until that are
            #       multiple tokens or that span multiple tokens correctly

            # TODO: extract to TokenizedLM?
            res = []

            def _collate(x):
                toks = self.tok_encode(x[0])
                return len(toks), x[0]

            re_ord = utils.Reorderer(requests, _collate)

            for context, request_args in tqdm(re_ord.get_reordered()):
                until = request_args["until"]
                if isinstance(until, str):
                    until = [until]

                if until:
                    (primary_until,) = self.tok_encode(until[0])
                else:
                    primary_until = None

                context_enc = torch.tensor(
                    [self.tok_encode(context)[self.max_gen_toks - self.max_length :]]
                ).to(self.device)

                max_gen_tokens = min(
                    self.max_gen_toks, request_args.get("max_length", self.max_gen_toks)
                )
                cont = self._model_generate(
                    context_enc, context_enc.shape[1] + max_gen_tokens, primary_until
                )

                s = self.tok_decode(cont[0].tolist()[context_enc.shape[1] :])

                for term in until:
                    s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), s)

                res.append(s)

            return re_ord.get_original(res)
    
except ImportError or ModuleNotFoundError:
    print("lm-eval==0.3.0 not install")