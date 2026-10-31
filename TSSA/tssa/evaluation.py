import numpy as np
import torch
from torch.utils.data import DataLoader
from .runtime import autocast


def retrieval_metrics(scores, img2txt, txt2img, labels=None):
    scores = np.asarray(scores)
    if scores.shape != (len(img2txt), len(txt2img)) or not np.isfinite(scores).all():
        raise ValueError('Invalid full-gallery score matrix')
    i_order = np.argsort(-scores, axis=1, kind='stable')
    i_inverse = np.argsort(i_order, axis=1, kind='stable')
    i_ranks = np.array([min(i_inverse[i, j] for j in img2txt[i]) for i in range(len(img2txt))])
    t_order = np.argsort(-scores.T, axis=1, kind='stable')
    t_inverse = np.argsort(t_order, axis=1, kind='stable')
    owners = np.array([txt2img[j] for j in range(len(txt2img))])
    t_ranks = t_inverse[np.arange(len(txt2img)), owners]
    def summarize(ir, tr):
        result = {f'{direction}_r{k}': 100 * float(np.mean(ranks < k))
                  for direction, ranks in [('i2t', ir), ('t2i', tr)] for k in (1, 5, 10)}
        result['i2t_mr'] = float(np.mean([result[f'i2t_r{k}'] for k in (1, 5, 10)]))
        result['t2i_mr'] = float(np.mean([result[f't2i_r{k}'] for k in (1, 5, 10)]))
        result['mR'] = (result['i2t_mr'] + result['t2i_mr']) / 2
        return result
    result = summarize(i_ranks, t_ranks)
    result.update(image_queries=len(i_ranks), text_queries=len(t_ranks))
    if labels is not None:
        labels = np.asarray(labels)
        per_class = {str(c): {**summarize(i_ranks[labels == c], t_ranks[labels[owners] == c]),
                              'image_queries': int((labels == c).sum()),
                              'text_queries': int((labels[owners] == c).sum())} for c in np.unique(labels)}
        result['per_class'] = per_class
        result['macro'] = {key: float(np.mean([row[key] for row in per_class.values()]))
                           for key in next(iter(per_class.values())) if key not in ['image_queries', 'text_queries']}
    return result


@torch.inference_mode()
def encode_dataset(model, dataset, tokenizer, device, cfg):
    model.eval()
    visual, textual = [], []
    loader = DataLoader(dataset, batch_size=cfg['eval_batch_size'], shuffle=False,
                        num_workers=cfg['num_workers'])
    for images, _ in loader:
        with autocast(device):
            visual.append(model.encode_image(images.to(device)).float().cpu())
    for start in range(0, len(dataset.texts), cfg['text_batch_size']):
        tokens = tokenizer(dataset.texts[start:start + cfg['text_batch_size']], padding=True,
                           truncation=True, max_length=cfg['max_length'], return_tensors='pt').to(device)
        with autocast(device):
            textual.append(model.encode_text(tokens['input_ids'], tokens['attention_mask']).float().cpu())
    return torch.cat(visual), torch.cat(textual)


def score_dataset(model, dataset, tokenizer, device, cfg):
    v, t = encode_dataset(model, dataset, tokenizer, device, cfg)
    v = torch.nn.functional.normalize(v, dim=-1)
    t = torch.nn.functional.normalize(t, dim=-1)
    scores = (v @ t.T).numpy()
    labels = [r['label'] for r in dataset.records]
    return scores, retrieval_metrics(scores, dataset.img2txt, dataset.txt2img, labels)
