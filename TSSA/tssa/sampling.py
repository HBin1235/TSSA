from __future__ import annotations
from collections import Counter
import hashlib
import json
import operator
import torch
from torch.utils.data import Sampler

class PairGroupedBatchSampler(Sampler[list[int]]):
    policy = 'pair_grouped'

    def __init__(self, labels, batch_size: int, seed: int):
        self.labels = tuple((operator.index(label) for label in labels))
        self.batch_size = operator.index(batch_size)
        self.seed = operator.index(seed)
        if not self.labels:
            raise ValueError('pair_grouped requires a nonempty dataset')
        if self.batch_size < 6 or self.batch_size % 2:
            raise ValueError('pair_grouped requires an even image batch size >= 6')
        if self.seed < 0:
            raise ValueError('seed must be nonnegative')
        self._by_class: dict[int, list[int]] = {}
        for index, label in enumerate(self.labels):
            self._by_class.setdefault(label, []).append(index)
        if any((len(indices) < 2 for indices in self._by_class.values())):
            raise ValueError('Every training class needs >= 2 distinct images; none will be excluded')
        full, remainder = divmod(len(self.labels), self.batch_size)
        self._sizes = [self.batch_size] * full + ([remainder] if remainder else [])
        self._min_triples = [size % 2 for size in self._sizes]
        self._max_triples = []
        for size in self._sizes:
            maximum = size // 3
            maximum -= (maximum - size % 2) % 2
            self._max_triples.append(maximum)
        total_triples = sum((len(indices) % 2 for indices in self._by_class.values()))
        if any((lo > hi for lo, hi in zip(self._min_triples, self._max_triples))) or not sum(self._min_triples) <= total_triples <= sum(self._max_triples) or (total_triples - sum(self._min_triples)) % 2:
            raise ValueError('Cannot preserve these batch sizes with whole same-class pairs/triples')
        self.epoch = 1
        self._cached_batches: tuple[tuple[int, ...], ...] | None = None

    def set_epoch(self, epoch: int):
        epoch = operator.index(epoch)
        if epoch < 1:
            raise ValueError('epoch is one-based and must be positive')
        if self.epoch != epoch:
            self._cached_batches = None
        self.epoch = epoch

    @staticmethod
    def _shuffle(values: list, generator: torch.Generator):
        return [values[index] for index in torch.randperm(len(values), generator=generator).tolist()]

    def _batches(self):
        if self._cached_batches is not None:
            return self._cached_batches
        seed_bytes = hashlib.blake2b(f'{self.seed}:{self.epoch}'.encode('ascii'), digest_size=8).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(seed_bytes, 'little') % 2 ** 63)
        pairs, triples = ([], [])
        for indices in self._by_class.values():
            indices = self._shuffle(indices, generator)
            if len(indices) % 2:
                triples.append(indices[:3])
                indices = indices[3:]
            pairs.extend((indices[start:start + 2] for start in range(0, len(indices), 2)))
        pairs, triples = (self._shuffle(pairs, generator), self._shuffle(triples, generator))
        triple_counts = self._min_triples.copy()
        slots = [batch for batch, (lo, hi) in enumerate(zip(self._min_triples, self._max_triples)) for _ in range((hi - lo) // 2)]
        slots = self._shuffle(slots, generator)
        extras = (len(triples) - sum(triple_counts)) // 2
        for batch in slots[:extras]:
            triple_counts[batch] += 2
        result, pair_offset, triple_offset = ([], 0, 0)
        for size, n_triples in zip(self._sizes, triple_counts):
            n_pairs = (size - 3 * n_triples) // 2
            groups = pairs[pair_offset:pair_offset + n_pairs] + triples[triple_offset:triple_offset + n_triples]
            indices = [index for group in groups for index in group]
            if len(indices) != size:
                raise RuntimeError('Pair/triple packing did not fill a batch')
            result.append(tuple(self._shuffle(indices, generator)))
            pair_offset += n_pairs
            triple_offset += n_triples
        if pair_offset != len(pairs) or triple_offset != len(triples):
            raise RuntimeError('Pair/triple packing left images unused')
        self._cached_batches = tuple(result)
        return self._cached_batches

    def __iter__(self):
        return (list(batch) for batch in self._batches())

    def __len__(self):
        return len(self._sizes)

    def summary(self):
        batches = self._batches()
        counts = [Counter((self.labels[index] for index in batch)) for batch in batches]
        singleton_classes = sum((n == 1 for count in counts for n in count.values()))
        payload = json.dumps(batches, separators=(',', ':')).encode('ascii')
        return {'policy': self.policy, 'epoch': self.epoch, 'seed': self.seed, 'images': len(self.labels), 'batches': len(batches), 'batch_sizes': list(map(len, batches)), 'min_images_per_present_class': min((n for count in counts for n in count.values())), 'singleton_class_batches': singleton_classes, 'class_image_counts': dict(Counter(self.labels)), 'class_batch_counts': {label: sum((label in count for count in counts)) for label in self._by_class}, 'schedule_sha256': hashlib.sha256(payload).hexdigest()}
