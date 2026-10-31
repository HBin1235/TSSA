from pathlib import Path
import contextlib
import json
import math
import random
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer
from .model import TSSA


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()


def build_optimizer(model, cfg):
    groups = {}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if name.startswith(('fine_projection', 'coarse_projection', 'text_projection', 'logit_scale')):
            lr = cfg['projection_learning_rate']
        elif name.startswith(('hsrp', 'tea')):
            lr = cfg['refinement_learning_rate']
        else:
            lr = cfg['learning_rate']
        decay = cfg['weight_decay'] if value.ndim >= 2 and not name.endswith('.bias') else 0.0
        groups.setdefault((lr, decay), []).append(value)
    return torch.optim.AdamW([{'params': params, 'lr': lr, 'weight_decay': decay}
                             for (lr, decay), params in groups.items()],
                            betas=tuple(cfg['betas']), eps=cfg['eps'])


def build_scheduler(optimizer, total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return (1 + math.cos(math.pi * progress)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def load_model(config, checkpoint, device):
    model = TSSA(config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval()


def tokenizer_for(config):
    return AutoTokenizer.from_pretrained(config['model']['bert_model'])
