import argparse
import numpy as np
from tssa.data import CropRetrievalEvalDataset
from tssa.evaluation import retrieval_metrics
from tssa.runtime import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output', default='outputs/evaluation.json')
    args = p.parse_args()
    dataset = CropRetrievalEvalDataset(args.annotations, '.', None)
    with np.load(args.scores, allow_pickle=False) as f:
        scores = f['scores']
    result = retrieval_metrics(scores, dataset.img2txt, dataset.txt2img,
                               [r['label'] for r in dataset.records])
    save_json(args.output, result)
    print(result['mR'])


if __name__ == '__main__':
    main()
