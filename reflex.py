"""Standalone text decision runtime. Modified model; not a text generator.

Apache-2.0. EmbeddingGemma 2 base model: Google DeepMind.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np


class Reflex:
    def __init__(self, model_root, seed=17, encoder_executable=None, head_path=None):
        self.root = Path(model_root)
        folder = self.root / f'seed-{seed}'
        with np.load(head_path or folder / 'head.npz', allow_pickle=False) as f:
            self.weights = {k: f[k] for k in f.files if k != 'metadata'}
        self.options = {}
        self.process = None
        if encoder_executable:
            self.process = subprocess.Popen([str(encoder_executable), str(folder / 'text-q8'), 'compiled-exact'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
            if not json.loads(self.process.stdout.readline()).get('ready'):
                raise RuntimeError('Encoder did not become ready')
        else:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoTokenizer, EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel
            torch.set_num_threads(2)
            directory = folder / 'text-bf16'
            config = EmbeddingGemma2TextConfig(**json.loads((directory/'config.json').read_text())['text_config'])
            config._attn_implementation = 'eager'
            self.model = EmbeddingGemma2TextModel(config).float().eval()
            source = load_file(str(directory / 'model.safetensors'))
            self.model.load_state_dict({k.removeprefix('language_model.'):v for k,v in source.items()}, strict=True)
            self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)

    def encode(self, texts):
        if self.process:
            self.process.stdin.write(json.dumps({'texts': texts})+'\n'); self.process.stdin.flush()
            result = json.loads(self.process.stdout.readline())
            if 'error' in result: raise RuntimeError(result['error'])
            return np.asarray(result['embeddings'], np.float32)
        import torch
        tokens = self.tokenizer(texts, padding=True, return_tensors='pt')
        if tokens['input_ids'].shape[1] > 192:
            raise ValueError('This research release was trained with at most 192 tokens per text.')
        with torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state.float()
            mask = tokens['attention_mask'].unsqueeze(-1)
            pooled = (hidden*mask).sum(1)/mask.sum(1)
            return torch.nn.functional.normalize(pooled, dim=-1).numpy()

    def logits(self, query, options):
        w=self.weights; x=np.asarray(query,np.float32)*math.sqrt(768); o=np.asarray(options,np.float32)
        a=np.tanh(x@w['query.0.weight'].T+w['query.0.bias'])
        a=a@w['query.2.weight'].T+w['query.2.bias']
        b=np.tanh(o*math.sqrt(768)@w['option.weight'].T+w['option.bias'])
        g=np.tanh(x@w['gate.0.weight'].T+w['gate.0.bias'])@w['gate.2.weight'].T+w['gate.2.bias']
        gate=1/(1+np.exp(-np.clip(g,-60,60)))
        return (o*query).sum(-1)/.08+gate*(a*b).sum(-1)/math.sqrt(128)

    def predict(self, instructions, state, options):
        if not isinstance(instructions,str) or not isinstance(state,str): raise TypeError('instructions/state must be text')
        if not 2<=len(options)<=32 or not all(isinstance(x,str) for x in options): raise ValueError('Supply 2–32 text options')
        if len(set(options))!=len(options): raise ValueError('Options must be distinct')
        start=time.perf_counter()
        missing=sorted(set(options)-self.options.keys())
        vectors=self.encode(['task: classification | query: '+instructions+' '+state]+missing)
        self.options.update(zip(missing,vectors[1:]))
        z=self.logits(vectors[0],np.asarray([self.options[o] for o in options],np.float32))
        p=np.exp(z-z.max());p/=p.sum();index=int(p.argmax())
        return {'index':index,'choice':options[index],'probabilities':p.tolist(),
                'probabilities_calibrated':False,'wall_ms':(time.perf_counter()-start)*1000}

    def close(self):
        if self.process:
            self.process.stdin.close()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=5)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-root',type=Path,default=Path(__file__).parent)
    p.add_argument('--seed',type=int,choices=[17,41],default=17)
    p.add_argument('--encoder',type=Path,help='Built reflex-embed Swift executable; otherwise CPU Torch reference')
    p.add_argument('--head',type=Path,help='Optional explicitly selected compatible head')
    p.add_argument('--request',required=True,help='JSON object with instructions, state, options')
    args=p.parse_args();model=Reflex(args.model_root,args.seed,args.encoder,args.head)
    try: print(json.dumps(model.predict(**json.loads(args.request)),ensure_ascii=False))
    finally:model.close()


if __name__=='__main__':main()
