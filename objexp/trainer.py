# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

from .model import ObjDetectModel


class ObjectnessTrainer(DetectionTrainer):
    """Detection trainer that builds :class:`ObjDetectModel` instead of the stock :class:`DetectionModel`."""

    def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True):
        """Return an ObjDetectModel with pretrained weights transferred where shapes match.

        Args:
            cfg (str | dict | None): Model configuration file path or dictionary.
            weights (torch.nn.Module | None): Pretrained model whose weights are loaded into the new model.
            verbose (bool): Whether to display model information.

        Returns:
            (ObjDetectModel): Detection model with an objectness branch.
        """
        model = self.set_model_names_for_load(
            ObjDetectModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model
