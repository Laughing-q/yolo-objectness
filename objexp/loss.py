# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class ObjectnessLoss(v8DetectionLoss):
    """Detection loss with a YOLOv5-style factorized objectness/classification training signal.

    Differences from :class:`v8DetectionLoss`:
    - The TaskAlignedAssigner ranks candidates with the combined score σ(cls) · σ(obj), matching inference confidence.
    - The classification BCE is restricted to positive anchors (P(class | object)); background rejection moves to the
      objectness branch.
    - An objectness BCE over all anchors supervises P(object) against the foreground mask.
    """

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        """Initialize the loss and append the objectness item to the loss names.

        Args:
            model (torch.nn.Module): Model whose head provides nc, reg_max and stride; must be de-parallelled.
            tal_topk (int): Number of top anchors per ground truth in the TaskAlignedAssigner.
            tal_topk2 (int | None): Optional second-stage top-k selection for one-to-one assignment.
        """
        super().__init__(model, tal_topk, tal_topk2)
        self.loss_names = self.loss_names + ("obj_loss",)

    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any], **assign_kwargs
    ) -> tuple:
        """Calculate the box, cls, dfl and obj losses and return them with the assigned targets.

        Args:
            preds (dict[str, torch.Tensor]): Raw head outputs with "boxes", "scores", "obj" and "feats" keys.
            batch (dict[str, Any]): Batch dict with image tensors and target annotations.
            **assign_kwargs (Any): Extra keyword arguments forwarded to the TaskAlignedAssigner.

        Returns:
            (tuple): Assigned target tuple, the (box, cls, dfl, obj) loss tensor, and a detached loss items dict.
        """
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, obj
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        pred_obj = preds["obj"].permute(0, 2, 1).contiguous()  # (bs, num_anchors, 1)
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            (pred_scores.sigmoid() * pred_obj.sigmoid()).detach(),  # combined score ranks like inference confidence
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            **assign_kwargs,
        )

        target_scores_sum = target_scores.sum().clamp_(min=1)  # on the device: no host sync

        # Cls loss on positive anchors only (P(class | object)), optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype)) * fg_mask.unsqueeze(-1)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Obj loss over all anchors (P(object))
        loss[3] = self.bce(pred_obj, fg_mask.unsqueeze(-1).to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss: zero on an empty foreground, and pred_distri stays in the graph either way
        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= getattr(self.hyp, "obj", 1.0)  # obj gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )  # loss(box, cls, dfl, obj)
