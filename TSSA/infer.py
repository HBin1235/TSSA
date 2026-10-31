import argparse
from pathlib import Path
import numpy as np
import torch
from tssa.data import build_transforms, CropRetrievalEvalDataset
from tssa.evaluation import score_dataset
from tssa.runtime import load_config, load_model, tokenizer_for, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/main.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output', default='outputs/inference')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = load_model(cfg, args.checkpoint, device)
    _, transform = build_transforms(cfg['train']['image_size'])
    dataset = CropRetrievalEvalDataset(args.annotations, cfg['data']['image_root'], transform)
    scores, metrics = score_dataset(model, dataset, tokenizer_for(cfg), device, cfg['train'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'scores.npz', scores=scores)
    save_json(out / 'metrics.json', metrics)
    save_json(out / 'mapping.json', {'image_ids': [r['image_id'] for r in dataset.records],
                                    'texts': dataset.texts, 'img2txt': dataset.img2txt, 'txt2img': dataset.txt2img})
    print(metrics['mR'])


if __name__ == '__main__':
    main()
