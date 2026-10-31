import torch
import torch.nn.functional as F


def objective(visual, textual, image_ids, text_image_ids, labels, text_labels,
              scale, lambda_ca=0.5, margin=0.2):
    with torch.autocast(device_type=visual.device.type, enabled=False):
        v, t = F.normalize(visual.float(), dim=-1), F.normalize(textual.float(), dim=-1)
        positive = image_ids[:, None].eq(text_image_ids[None, :])
        if not positive.any(1).all() or not positive.sum(0).eq(1).all():
            raise ValueError('Each text must have one image owner; each image needs paired texts')
        owner = positive.long().argmax(0)
        if not torch.equal(labels[owner], text_labels):
            raise ValueError('Paired image and text labels must agree')
        logits = scale.float() * (v @ t.T)
        pos_logits = logits.masked_fill(~positive, -torch.inf)
        i2t = (torch.logsumexp(logits, 1) - torch.logsumexp(pos_logits, 1)).mean()
        t2i = (torch.logsumexp(logits, 0) - torch.logsumexp(pos_logits, 0)).mean()
        contrastive = (i2t + t2i) / 2
        classes = labels.unique()
        if len(classes) < 2:
            return contrastive, {'info_nce': float(contrastive.detach()), 'center_margin': 0.0}
        pv = F.normalize(torch.stack([v[labels == c].mean(0) for c in classes]), dim=-1)
        pt = F.normalize(torch.stack([t[text_labels == c].mean(0) for c in classes]), dim=-1)
        similarities = pv @ pt.T
        positive_center = similarities.diag()[:, None]
        off_diagonal = ~torch.eye(len(classes), device=v.device, dtype=torch.bool)
        forward = F.relu(margin + similarities - positive_center)[off_diagonal]
        backward = F.relu(margin + similarities.T - positive_center)[off_diagonal]
        center = (forward.sum() + backward.sum()) / (2 * len(classes) * (len(classes) - 1))
        return contrastive + lambda_ca * center, {'info_nce': float(contrastive.detach()),
                                                'center_margin': float(center.detach())}
