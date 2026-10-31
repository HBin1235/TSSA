import argparse
import copy
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from tssa.data import build_transforms, CropRetrievalTrainDataset, CropRetrievalEvalDataset, collate_train
from tssa.model import TSSA
from tssa.losses import objective
from tssa.sampling import PairGroupedBatchSampler
from tssa.runtime import load_config, save_json, seed_all, seed_worker, autocast, build_optimizer, build_scheduler, tokenizer_for
from tssa.evaluation import score_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/main.yaml')
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume')
    p.add_argument('--test-after-training', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg['train']['seed'] = args.seed
    tc, lc, dc = cfg['train'], cfg['loss'], cfg['data']
    seed_all(tc['seed'])
    device = torch.device(args.device)
    out = Path(cfg['output_dir']) / f'seed{tc["seed"]}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'checkpoint_last.pt').exists() and not args.resume:
        raise FileExistsError('Use --resume or choose an unused output_dir')
    save_json(out / 'config.json', cfg)
    train_tf, eval_tf = build_transforms(tc['image_size'])
    train = CropRetrievalTrainDataset(dc['train_json'], dc['image_root'], train_tf, tc['captions_per_image'])
    val = CropRetrievalEvalDataset(dc['val_json'], dc['image_root'], eval_tf)
    train_ids = {r['image_id'] for r in train.records}
    if train_ids.intersection(r['image_id'] for r in val.records):
        raise ValueError('Training and validation image identities overlap')
    sampler = PairGroupedBatchSampler([r['label'] for r in train.records], tc['batch_size'], tc['seed'])
    generator = torch.Generator().manual_seed(tc['seed'])
    loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate_train,
                        num_workers=tc['num_workers'], worker_init_fn=seed_worker,
                        generator=generator, pin_memory=device.type == 'cuda')
    model = TSSA(cfg).to(device)
    tokenizer = tokenizer_for(cfg)
    optimizer = build_optimizer(model, tc)
    scheduler = build_scheduler(optimizer, len(loader) * tc['epochs'], len(loader) * tc['warmup_epochs'])
    start, best = 1, -1.
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['config'] != cfg:
            raise ValueError('Resume configuration differs from the saved run')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best = state['epoch'] + 1, state['best_val_mR']
        import random
        import numpy as np
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        generator.set_state(state['rng']['loader'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    for epoch in range(start, tc['epochs'] + 1):
        sampler.set_epoch(epoch)
        model.train()
        running = 0.
        weight = lc['lambda_ca'] * min(1., (epoch - 1) / max(1, lc['warmup_epochs'] - 1))
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            tokens = tokenizer(batch['captions'], padding=True, truncation=True,
                               max_length=tc['max_length'], return_tensors='pt').to(device)
            with autocast(device):
                v, t = model(batch['images'].to(device), tokens['input_ids'], tokens['attention_mask'])
                loss, _ = objective(v, t, batch['image_ids'].to(device), batch['text_image_ids'].to(device),
                                    batch['image_labels'].to(device), batch['text_labels'].to(device),
                                    model.contrastive_scale(), weight, lc['margin'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), tc['max_grad_norm'])
            if not torch.isfinite(grad):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
        _, metrics = score_dataset(model, val, tokenizer, device, tc)
        improved = metrics['mR'] > best
        best = max(best, metrics['mR'])
        import random
        import numpy as np
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch': torch.get_rng_state(), 'loader': generator.get_state(),
               'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
        state = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 'config': copy.deepcopy(cfg), 'epoch': epoch, 'best_val_mR': best}
        if improved:
            torch.save(state, out / 'checkpoint_best.pt')
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng)
        torch.save(state, out / 'checkpoint_last.pt')
        record = {'epoch': epoch, 'loss': running / len(loader), 'lambda_ca': weight, 'val': metrics}
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    if args.test_after_training:
        state = torch.load(out / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        test = CropRetrievalEvalDataset(dc['test_json'], dc['image_root'], eval_tf)
        test_ids = {r['image_id'] for r in test.records}
        if test_ids.intersection(train_ids | {r['image_id'] for r in val.records}):
            raise ValueError('Test image identities overlap another split')
        scores, metrics = score_dataset(model, test, tokenizer, device, tc)
        import numpy as np
        np.savez_compressed(out / 'test_scores.npz', scores=scores)
        save_json(out / 'test_metrics.json', {'best_epoch': state['epoch'], 'metrics': metrics})


if __name__ == '__main__':
    main()
