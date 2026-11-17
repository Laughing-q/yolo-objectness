# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import initialize_weights

from .head import ObjDetect
from .loss import ObjectnessLoss


class ObjDetectModel(DetectionModel):
    """YOLO detection model whose stock head is swapped for :class:`ObjDetect` and trained with :class:`ObjectnessLoss`."""

    def __init__(self, cfg: str | dict = "yolo26.yaml", ch: int = 3, nc: int | None = None, verbose: bool = True):
        """Build the stock model from its YAML, then replace the head with the objectness variant.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int | None): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg, ch, nc, verbose)
        m = self.model[-1]
        head = ObjDetect(
            nc=m.nc,
            reg_max=m.reg_max,
            end2end=getattr(m, "one2one_cv2", None) is not None,
            ch=[b[0].conv.in_channels for b in m.cv2],
        )
        head.stride = m.stride
        head.f, head.i, head.type = m.f, m.i, m.type  # parse_model metadata read by _predict_once
        head.np = sum(x.numel() for x in head.parameters())
        head.bias_init()
        initialize_weights(head)
        self.model[-1] = head

    def init_criterion(self):
        """Initialize the objectness loss, wrapped in E2ELoss for dual-head models.

        Returns:
            (ObjectnessLoss | E2ELoss): Loss criterion matching the head layout.
        """
        return (
            E2ELoss(self, ObjectnessLoss)
            if getattr(self.model[-1], "one2one_cv2", None) is not None
            else ObjectnessLoss(self)
        )
