from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def build_transforms(image_size: int=224):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    train_transform = transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC), transforms.ToTensor(), normalize])
    return (train_transform, eval_transform)

def _load_json(path: str | Path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)

def clean_caption(caption: str) -> str:
    return ' '.join(caption.strip().split())

def _caption_list(value) -> list[str]:
    captions = [value] if isinstance(value, str) else list(value)
    return [clean_caption(caption) for caption in captions]

class CropRetrievalTrainDataset(Dataset):

    def __init__(self, annotation_file: str | Path, image_root: str | Path, transform, captions_per_image: int=3):
        raw_records = _load_json(annotation_file)
        self.image_root = Path(image_root)
        self.transform = transform
        self.captions_per_image = int(captions_per_image)
        if self.captions_per_image < 1:
            raise ValueError('captions_per_image must be positive')
        caption_group_lookup: dict[str, int] = {}
        grouped: dict[int, dict] = {}
        image_order: list[int] = []
        for row in raw_records:
            image_id = int(row['image_id'])
            label = int(row['label'])
            image = str(row['image'])
            if image_id not in grouped:
                grouped[image_id] = {'image': image, 'captions': [], 'caption_group_ids': [], 'image_id': image_id, 'label': label}
                image_order.append(image_id)
            else:
                current = grouped[image_id]
                if current['image'] != image or int(current['label']) != label:
                    raise ValueError(f'Inconsistent rows for image_id={image_id}')
            record = grouped[image_id]
            for caption in _caption_list(row['caption']):
                group_id = caption_group_lookup.setdefault(caption, len(caption_group_lookup))
                record['captions'].append(caption)
                record['caption_group_ids'].append(group_id)
        wrong = [(image_id, len(grouped[image_id]['captions'])) for image_id in image_order if len(grouped[image_id]['captions']) != self.captions_per_image]
        if wrong:
            preview = ', '.join((f'{image_id}:{count}' for image_id, count in wrong[:8]))
            raise ValueError(f'Every training image must retain exactly {self.captions_per_image} paired captions; mismatches: {preview}')
        self.records = [grouped[image_id] for image_id in image_order]
        self.num_images = len(self.records)
        self.num_captions = sum((len(record['captions']) for record in self.records))
        self.num_unique_captions = len(caption_group_lookup)
        self.classes = sorted({int(record['label']) for record in self.records})
        expected_classes = list(range(len(self.classes)))
        if self.classes != expected_classes:
            raise ValueError(f'Training labels must be contiguous and zero-based; found {self.classes}')
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.image_root / record['image']
        with Image.open(image_path) as source:
            image = self.transform(source.convert('RGB'))
        return {'image': image, 'captions': list(record['captions']), 'caption_group_ids': list(record['caption_group_ids']), 'image_id': int(record['image_id']), 'label': int(record['label'])}

def collate_train(batch: Iterable[dict]):
    items = list(batch)
    if not items:
        raise ValueError('Training batch must not be empty')
    captions_per_image = len(items[0]['captions'])
    if captions_per_image < 1:
        raise ValueError('Each training image must have at least one caption')
    if any((len(item['captions']) != captions_per_image for item in items)):
        raise ValueError('All images in a training batch must have the same caption count')
    image_ids = torch.tensor([item['image_id'] for item in items], dtype=torch.long)
    labels = torch.tensor([item['label'] for item in items], dtype=torch.long)
    captions: list[str] = []
    text_image_ids: list[int] = []
    text_labels: list[int] = []
    caption_group_ids: list[int] = []
    for item in items:
        captions.extend(item['captions'])
        text_image_ids.extend([int(item['image_id'])] * captions_per_image)
        text_labels.extend([int(item['label'])] * captions_per_image)
        caption_group_ids.extend((int(value) for value in item['caption_group_ids']))
    return {'images': torch.stack([item['image'] for item in items]), 'captions': captions, 'image_ids': image_ids, 'image_labels': labels, 'text_image_ids': torch.tensor(text_image_ids, dtype=torch.long), 'text_labels': torch.tensor(text_labels, dtype=torch.long), 'text_caption_group_ids': torch.tensor(caption_group_ids, dtype=torch.long), 'captions_per_image': captions_per_image}

class CropRetrievalEvalDataset(Dataset):

    def __init__(self, annotation_file: str | Path, image_root: str | Path, transform):
        self.records = _load_json(annotation_file)
        self.image_root = Path(image_root)
        self.transform = transform
        self.texts: list[str] = []
        self.img2txt: dict[int, list[int]] = {}
        self.txt2img: dict[int, int] = {}
        for image_index, record in enumerate(self.records):
            self.img2txt[image_index] = []
            for caption in _caption_list(record['caption']):
                text_index = len(self.texts)
                self.texts.append(caption)
                self.img2txt[image_index].append(text_index)
                self.txt2img[text_index] = image_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        image_path = self.image_root / self.records[index]['image']
        with Image.open(image_path) as source:
            image = self.transform(source.convert('RGB'))
        return (image, index)
