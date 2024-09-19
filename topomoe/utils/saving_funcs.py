import os
from typing import Any

import torch
import torchvision


class AbstractSaver:
    def __init__(self, extension: str, folder: str, save_id: bool = False):
        self.folder = folder
        self._id = 0
        self.extension = extension
        self.folder = "{}/{}_{}".format(
            self.folder, self.extension, self.__class__.__name__
        )
        self.save_id = save_id

    def _get_mkdir_path(self, *path):
        path = [str(folder) for folder in path]
        to_join = "_".join(path)
        child = "{}_{}".format(self._id, to_join) if self.save_id else to_join
        par = os.path.join("desktop", self.folder)
        os.makedirs(par, exist_ok=True)
        return "{}/{}.{}".format(par, child, self.extension)

    def save(self, result: torch.Tensor, *path):
        self.save_function(result, self._get_mkdir_path(*path))
        self._id += 1

    def save_function(self, result: Any, path: str):
        raise NotImplementedError


class ImageSaver(AbstractSaver):
    def save_function(self, result: Any, path: str):
        torchvision.utils.save_image(result, path, nrow=1)

    @staticmethod
    def get_nrow() -> torch.tensor:
        bs = torch.arange(1, 1000)
        p = bs.view(1, -1).repeat(len(bs), 1)
        q = (bs.view(-1, 1) / p).floor().int()
        feasible = (p * q) == bs.view(-1, 1)
        sm = p + q
        sm[feasible == False] = (len(bs) + 1) * 10
        return p[torch.arange(len(bs)), sm.argmin(dim=-1)]

    def __init__(self, folder: str, save_id: bool = False):
        super().__init__("png", folder, save_id)
        self.nrow = self.get_nrow()


class TensorSaver(AbstractSaver):
    def save_function(self, result: Any, path: str):
        torch.save(result, path)

    def __init__(self, folder: str, save_id: bool = False):
        super().__init__("pth", folder, save_id)


class ExperimentSaver(AbstractSaver):
    def save(self, result: torch.Tensor, *path):
        self.image.save(result, *path)
        if not self.disk_saver:
            self.tensor.save(result, *path)

    def __init__(self, folder: str, save_id: bool = False, disk_saver: bool = False):
        super().__init__("none", folder, save_id)
        self.image = ImageSaver(folder=folder, save_id=save_id)
        self.disk_saver = disk_saver
        self.tensor = TensorSaver(folder=folder, save_id=save_id)


def new_init(
    size: int,
    device: str,
    batch_size: int = 1,
    last: torch.nn = None,
    padding: int = -1,
    zero: bool = False,
) -> torch.nn:
    output = (
        torch.rand(size=(batch_size, 3, size, size))
        if not zero
        else torch.zeros(size=(batch_size, 3, size, size))
    )
    output = output.to(device)
    if last is not None:
        big_size = size if padding == -1 else size - padding
        up = torch.nn.Upsample(
            size=(big_size, big_size), mode="bilinear", align_corners=False
        ).to(device)
        scaled = up(last)
        cx = (output.patch_size(-1) - big_size) // 2
        output[:, :, cx : cx + big_size, cx : cx + big_size] = scaled
    output = output.detach().clone()
    output.requires_grad_()
    return output
