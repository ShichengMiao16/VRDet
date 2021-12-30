import torch
import torch.nn.functional as F

from .base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import poly2hbb
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class BinCoder(BaseBBoxCoder):
    def __init__(self, num_bins):
        super(BinCoder, self).__init__()
        self.num_bins = num_bins

    def encode(self, polys):
        """Get bin classification and offset regression targets during training.

        Args:
            polys (torch.Tensor): ground truth boxes in polygon format
            (x1, y1, x2, y2, x3, y3, x4, y4). Shape (n, 8).

        Returns:
            tuple[Tensor]: (bin_cls_targets, bin_cls_weights, bin_offset_targets, bin_offset_weights)
                - bin_cls_targets: bin classification targets. Shape (n, 4*#bin).
                - bin_cls_weights: bin classification weights. Shape (n, 4*#bin).
                - bin_offset_targets: bin offset regression targets. Shape (n, 4).
                - bin_offset_weights: bin offset regression weights. Shape (n, 4).
        """
        assert polys.size(1) == 8
        min_x, min_x_idx = polys[:, 0::2].min(1)
        max_x, max_x_idx = polys[:, 0::2].max(1)
        min_y, min_y_idx = polys[:, 1::2].min(1)
        max_y, max_y_idx = polys[:, 1::2].max(1)
        hbboxes = torch.stack([min_x, min_y, max_x, max_y], dim=1)

        (bin_w, bin_h, t_bins, b_bins, l_bins, r_bins) = generate_bins(hbboxes, self.num_bins)

        polys = polys.view(-1, 4, 2)
        num_polys = polys.size(0)
        polys_ordered = torch.zeros_like(polys)
        polys_ordered[:, 0] = polys[range(num_polys), min_y_idx]
        polys_ordered[:, 1] = polys[range(num_polys), max_x_idx]
        polys_ordered[:, 2] = polys[range(num_polys), max_y_idx]
        polys_ordered[:, 3] = polys[range(num_polys), min_x_idx]

        t_x = polys_ordered[:, 0, 0]
        r_y = polys_ordered[:, 1, 1]
        b_x = polys_ordered[:, 2, 0]
        l_y = polys_ordered[:, 3, 1]

        # generate offset targets and weights
        # offsets from gts to bin centerlines
        t_offsets_ = (t_x[:, None] - t_bins) / bin_w[:, None]
        b_offsets_ = (b_x[:, None] - b_bins) / bin_w[:, None]
        l_offsets_ = (l_y[:, None] - l_bins) / bin_h[:, None]
        r_offsets_ = (r_y[:, None] - r_bins) / bin_h[:, None]

        # select the nearest bins
        _, t_bin_labels = t_offsets_.abs().min(1)
        _, b_bin_labels = b_offsets_.abs().min(1)
        _, l_bin_labels = l_offsets_.abs().min(1)
        _, r_bin_labels = r_offsets_.abs().min(1)

        t_offsets = t_offsets_.gather(1, t_bin_labels[:, None])
        b_offsets = b_offsets_.gather(1, b_bin_labels[:, None])
        l_offsets = l_offsets_.gather(1, l_bin_labels[:, None])
        r_offsets = r_offsets_.gather(1, r_bin_labels[:, None])

        # generate offset targets
        bin_offset_targets = torch.cat((t_offsets, b_offsets,
                                        l_offsets, r_offsets), dim=-1)
        
        # generate offset weights
        bin_offset_weights = torch.ones(bin_offset_targets.size())


        # generate bin labels and weights
        # generate bin labels
        labels = torch.stack((t_bin_labels, b_bin_labels,
                              l_bin_labels, r_bin_labels), dim=1)
        bin_cls_targets = F.one_hot(labels.view(-1), 
                               self.num_bins).view(labels.size(0), -1).float()

        # generate bin weights
        bin_cls_t_weights = (t_offsets_.abs() <= 0.5).float()
        bin_cls_b_weights = (b_offsets_.abs() <= 0.5).float()
        bin_cls_l_weights = (l_offsets_.abs() <= 0.5).float()
        bin_cls_r_weights = (r_offsets_.abs() <= 0.5).float()
        bin_cls_weights = torch.cat((bin_cls_t_weights, bin_cls_b_weights,
                                     bin_cls_l_weights, bin_cls_r_weights), dim=-1)
        bin_cls_weights = (~((bin_cls_weights == 1) & (bin_cls_targets == 0))).float()
                                     
        return bin_cls_targets, bin_cls_weights, bin_offset_targets, bin_offset_weights

    def decode(self, hbboxes, bin_cls_preds, bin_offset_preds):
        """Decode bin_cls_preds and bin_offset_preds to get the predicted polys.

        Args:
            hbboxes (torch.Tensor): horizontal bboxes. Shape (n, 4*#class).
            bin_cls_preds (torch.Tensor): bin classification predictionis.
                Shape (n, 4*#bin*#class).
            bin_offset_preds (torch.Tensor): bin offset predictions.
                Shape (n, 4*#class).

        Returns:
            polys: decoded polygons. Shape (n, 8*#class).
        """
        # decode bin_cls_preds
        bin_cls_preds = bin_cls_preds.view(hbboxes.size(0), -1, 4*self.num_bins)
        t_bin_cls_preds = bin_cls_preds[..., 0:self.num_bins]
        b_bin_cls_preds = bin_cls_preds[..., self.num_bins:2*self.num_bins]
        l_bin_cls_preds = bin_cls_preds[..., 2*self.num_bins:3*self.num_bins]
        r_bin_cls_preds = bin_cls_preds[..., 3*self.num_bins:4*self.num_bins]

        t_bin_cls_scores = F.softmax(t_bin_cls_preds, dim=-1)
        b_bin_cls_scores = F.softmax(b_bin_cls_preds, dim=-1)
        l_bin_cls_scores = F.softmax(l_bin_cls_preds, dim=-1)
        r_bin_cls_scores = F.softmax(r_bin_cls_preds, dim=-1)

        _, t_bin_inds = t_bin_cls_scores.max(-1)
        _, b_bin_inds = b_bin_cls_scores.max(-1)
        _, l_bin_inds = l_bin_cls_scores.max(-1)
        _, r_bin_inds = r_bin_cls_scores.max(-1)

        hx1 = hbboxes[:, 0::4]
        hy1 = hbboxes[:, 1::4]
        hx2 = hbboxes[:, 2::4]
        hy2 = hbboxes[:, 3::4]
        hw = hx2 - hx1
        hh = hy2 - hy1
        bin_w = (hw / self.num_bins)
        bin_h = (hh / self.num_bins)

        # generate bin centerline coords
        t_bins = (hx1.view(-1, 1) + (0.5 + torch.arange(0, 
            self.num_bins).to(hbboxes).float())[None, :] * bin_w.view(-1, 1)).view(hbboxes.size(0), -1, self.num_bins)
        b_bins = (hx2.view(-1, 1) - (0.5 + torch.arange(0, 
            self.num_bins).to(hbboxes).float())[None, :] * bin_w.view(-1, 1)).view(hbboxes.size(0), -1, self.num_bins)
        l_bins = (hy1.view(-1, 1) + (0.5 + torch.arange(0, 
            self.num_bins).to(hbboxes).float())[None, :] * bin_h.view(-1, 1)).view(hbboxes.size(0), -1, self.num_bins)
        r_bins = (hy2.view(-1, 1) - (0.5 + torch.arange(0, 
            self.num_bins).to(hbboxes).float())[None, :] * bin_h.view(-1, 1)).view(hbboxes.size(0), -1, self.num_bins)

        # decode bin_offset_preds
        bin_offset_preds = bin_offset_preds.view(hbboxes.size(0), -1, 4)
        t_bin_offset_preds = bin_offset_preds[..., 0]
        b_bin_offset_preds = bin_offset_preds[..., 1]
        l_bin_offset_preds = bin_offset_preds[..., 2]
        r_bin_offset_preds = bin_offset_preds[..., 3]

        pred_tx = t_bins.gather(2, t_bin_inds[..., None]).squeeze(2) + t_bin_offset_preds * bin_w
        pred_bx = b_bins.gather(2, b_bin_inds[..., None]).squeeze(2) + b_bin_offset_preds * bin_w
        pred_ly = l_bins.gather(2, l_bin_inds[..., None]).squeeze(2) + l_bin_offset_preds * bin_h
        pred_ry = r_bins.gather(2, r_bin_inds[..., None]).squeeze(2) + r_bin_offset_preds * bin_h

        polys = torch.stack((pred_tx, hy1, hx2, pred_ry,
                             pred_bx, hy2, hx1, pred_ly), dim=-1)
        polys = polys.flatten(1)

        return polys


def generate_bins(hbboxes, num_bins):

    hx1 = hbboxes[..., 0]
    hy1 = hbboxes[..., 1]
    hx2 = hbboxes[..., 2]
    hy2 = hbboxes[..., 3]

    hw = hx2 - hx1
    hh = hy2 - hy1

    bin_w = hw / num_bins
    bin_h = hh / num_bins

    # top bins
    t_bins = hx1[:, None] + (0.5 + torch.arange(0, 
        num_bins).to(hbboxes).float())[None, :] * bin_w[:, None]

    # bottom bins
    b_bins = hx2[:, None] - (0.5 + torch.arange(0,
        num_bins).to(hbboxes).float())[None, :] * bin_w[:, None]

    # left bins
    l_bins = hy1[:, None] + (0.5 + torch.arange(0, 
        num_bins).to(hbboxes).float())[None, :] * bin_h[:, None]

    # right bins
    r_bins = hy2[:, None] - (0.5 + torch.arange(0,
        num_bins).to(hbboxes).float())[None, :] * bin_h[:, None]

    return bin_w, bin_h, t_bins, b_bins, l_bins, r_bins


@BBOX_CODERS.register_module()
class RatioCoder(BaseBBoxCoder):

    def encode(self, polys):
        assert polys.size(1) == 8
        hbboxes = poly2hbb(polys)
        h_areas = (hbboxes[:, 2] - hbboxes[:, 0]) * \
                (hbboxes[:, 3] - hbboxes[:, 1])

        polys = polys.view(polys.size(0), 4, 2)
        areas = polys.new_zeros(polys.size(0))
        for i in range(4):
            areas += 0.5 * (polys[:, i, 0] * polys[:, (i+1)%4, 1] -
                            polys[:, (i+1)%4, 0] * polys[:, i, 1])
        areas = torch.abs(areas)

        ratios = areas / h_areas
        return ratios[:, None]

    def decode(self):
        raise NotImplementedError