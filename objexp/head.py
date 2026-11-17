# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect


class ObjDetect(Detect):
    """YOLO Detect head with an added YOLOv5-style objectness branch.

    Extends :class:`Detect` with per-level ``cv4``/``one2one_cv4`` objectness convolutions and returns an extra
    ``"obj"`` key in the training prediction dict. Inference confidence is the product ``σ(cls) · σ(obj)``, while the
    output tensor keeps the stock ``(B, 4 + nc, N)`` layout so downstream postprocess/NMS/validator code is unchanged.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: list[int] | tuple[int, ...] = ()):
        """Initialize the detection head and its parallel objectness branch.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (list[int] | tuple[int, ...]): Channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))  # cls branch hidden width
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, 1, 1)) for x in ch)
        if getattr(self, "one2one_cv2", None) is not None:
            self.one2one_cv4 = deepcopy(self.cv4)

    @property
    def one2many(self) -> dict[str, nn.Module]:
        """Return the one-to-many head components including the objectness branch."""
        return {**super().one2many, "obj_head": self.cv4}

    @property
    def one2one(self) -> dict[str, nn.Module]:
        """Return the one-to-one head components including the objectness branch."""
        return {**super().one2one, "obj_head": self.one2one_cv4}

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
        obj_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Concatenate and return predicted boxes, class probabilities and objectness.

        Args:
            x (list[torch.Tensor]): Feature maps from each detection level.
            box_head (nn.Module | None): Box regression convolutions.
            cls_head (nn.Module | None): Classification convolutions.
            obj_head (nn.Module | None): Objectness convolutions.

        Returns:
            (dict[str, torch.Tensor]): Prediction dict with "boxes", "scores", "obj" and "feats" keys.
        """
        out = super().forward_head(x, box_head, cls_head)
        if obj_head is not None and out:
            bs = x[0].shape[0]  # batch size
            out["obj"] = torch.cat([obj_head[i](x[i]).view(bs, 1, -1) for i in range(self.nl)], dim=-1)
        return out

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode boxes and combine class with objectness into the confidence channels.

        Args:
            x (dict[str, torch.Tensor]): Dictionary of predictions from detection layers.

        Returns:
            (torch.Tensor): Decoded predictions of shape (B, 4 + nc, num_anchors) with conf = σ(cls) · σ(obj).
        """
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid() * x["obj"].sigmoid()), 1)

    def bias_init(self):
        """Initialize detection head biases, including the objectness branch."""
        super().bias_init()
        for heads in (self.cv4, getattr(self, "one2one_cv4", ())):
            for i, h in enumerate(heads):
                h[-1].bias.data[:] = math.log(5 / (640 / self.stride[i]) ** 2)  # obj (1 channel, 5 objects, 640 img)
