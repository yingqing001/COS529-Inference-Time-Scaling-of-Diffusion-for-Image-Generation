from PIL import Image
import io
import numpy as np
import torch


def jpeg_incompressibility():
    def _fn(images, prompts=None, metadata=None):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        # return np.array(sizes), {}
        return np.array(sizes)

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts=None, metadata=None):
        # rew, meta = jpeg_fn(images, prompts, metadata)
        rew = jpeg_fn(images, prompts, metadata)
        # return -rew, meta
        return -rew

    return _fn


def aesthetic_score():
    from aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts=None, metadata=None):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        # return scores, {}
        return scores

    return _fn

