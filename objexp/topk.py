# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from functools import partial

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, nms, ops


class TopkPredictor(DetectionPredictor):
    """Detection predictor that attaches top-k class scores to every predicted object.

    Beyond the stock ``[x1, y1, x2, y2, conf, cls]`` output, each :class:`Results` carries ``topk_classes`` and
    ``topk_scores`` tensors of shape (num_boxes, k) — the k highest class scores of each kept detection, sorted
    descending (column 0 mirrors the final ``boxes.cls``/``boxes.conf``). ``k`` is adjustable via the ``topk`` class or
    constructor argument.

    Works with stock and objectness-branch models on the external-NMS path (``nms=None``/``True``). End-to-end
    (NMS-free, ``nms=False``) outputs are already reduced to top-1 before post-processing and fall back to stock
    behavior with a warning.

    Attributes:
        topk (int): Number of class scores returned per predicted object.

    Examples:
        >>> from ultralytics import YOLO
        >>> from objexp.topk import TopkPredictor
        >>> model = YOLO("best.pt")
        >>> results = model.predict("img.jpg", predictor=TopkPredictor)  # top-5 by default
        >>> results[0].topk_classes, results[0].topk_scores  # (N, 5) tensors per image
    """

    topk = 5

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, topk: int | None = None):
        """Initialize the predictor with an adjustable number of per-object class scores.

        Args:
            cfg (str | Path | dict | None): Configuration file path or dictionary.
            overrides (dict | None): Configuration overrides.
            _callbacks (dict | None): Callback overrides.
            topk (int | None): Number of class scores per object; defaults to the class attribute ``topk``.
        """
        super().__init__(cfg, overrides, _callbacks)
        if topk is not None:
            self.topk = topk

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Run NMS and return Results carrying top-k class scores per kept detection.

        Args:
            preds (torch.Tensor | tuple): Raw model predictions, ``(B, 4 + nc, N)`` for the external-NMS path.
            img (torch.Tensor): Preprocessed input images.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments forwarded to ``construct_results``.

        Returns:
            (list[Results]): Results objects with ``topk_classes``/``topk_scores`` of shape (num_boxes, k) attached.
        """
        if getattr(self.model, "end2end", False):
            LOGGER.warning("TopkPredictor: end-to-end outputs carry top-1 scores only; returning stock results.")
            return super().postprocess(preds, img, orig_imgs, **kwargs)

        p = preds[0] if isinstance(preds, (list, tuple)) else preds
        nc = len(self.model.names)
        k = min(self.topk, nc)
        topk_scores, topk_classes = p[:, 4 : 4 + nc].topk(k, dim=1)  # (B, k, N)

        save_feats = getattr(self, "_feats", None) is not None
        preds, idxs = nms.non_max_suppression(
            p,
            self.args.conf,
            kwargs.pop("iou", self.args.iou),
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0,
            end2end=False,
            rotated=self.args.task == "obb",
            return_idxs=True,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        obj_feats = self.get_obj_feats(self._feats, idxs) if save_feats else None
        results = self.construct_results(preds, img, orig_imgs, **kwargs)
        for b, (r, idx) in enumerate(zip(results, idxs)):
            r.topk_classes = topk_classes[b, :, idx].T  # (num_boxes, k)
            r.topk_scores = topk_scores[b, :, idx].T
            if save_feats:
                r.feats = obj_feats[b]
        return results


def predict_topk(model, source, k: int = 5, **kwargs):
    """Run prediction with top-k class scores attached to each predicted object.

    Args:
        model (ultralytics.YOLO): Detection model to run.
        source (str | Path | np.ndarray | torch.Tensor | list): Image source(s) accepted by ``model.predict``.
        k (int): Number of class scores to return per predicted object.
        **kwargs (Any): Forwarded to ``model.predict`` (e.g. ``conf``, ``imgsz``).

    Returns:
        (list[Results]): Prediction results with ``topk_classes``/``topk_scores`` of shape (num_boxes, k).
    """
    return model.predict(source, predictor=partial(TopkPredictor, topk=k), **kwargs)
