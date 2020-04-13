
"""Generic RoI Extractor."""

from __future__ import division

from torch import nn
from copy import deepcopy

from mmdet.core import force_fp32
from mmdet.models.plugins.non_local import NonLocal2D
from mmdet.models.plugins.generalized_attention import GeneralizedAttention
from mmdet.models.utils import ConvModule
from ..registry import ROI_EXTRACTORS
from .single_level import SingleRoIExtractor


class NoProcess(object):
    """Apply identity function."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


def SequentialModule(modules):
    """Apply a sequential module."""
    nn_modules = []
    for mc in modules:
        type_ = mc.pop('type')
        nn_modules.append(load_processing_model(type_, mc))
    return nn.Sequential(*nn_modules)


# Models supported by GRoIE
models = {
    'NonLocal2D': NonLocal2D,
    'GeneralizedAttention': GeneralizedAttention,
    'ConvModule': ConvModule,
    'Sequential': SequentialModule,
    'ReLU': nn.ReLU,
}


def load_processing_model(model_type, config):
    """Load a specific module."""
    return models.get(model_type, NoProcess)(**config)


@ROI_EXTRACTORS.register_module
class SumGenericRoiExtractor(SingleRoIExtractor):
    """Extract RoI features from all summed feature maps levels.

    Args:
        pre_conf (dict): Specify pre-processing modules.
        post_conf (dict): Specify post-processing modules.
        ... (see SingleRoIExtractor for other parameters)
    """

    def __init__(self, **kwargs):
        kwargs = deepcopy(kwargs)

        post_conf = kwargs.pop('post_conf')
        pre_conf = kwargs.pop('pre_conf')

        pre_type = pre_conf.pop('type')
        post_type = post_conf.pop('type')

        super(SumGenericRoiExtractor, self).__init__(**kwargs)

        # build pre/post processing modules
        self.post_conv = load_processing_model(post_type, post_conf)
        self.pre_conv = load_processing_model(pre_type, pre_conf)
        self.relu = nn.ReLU(inplace=False)

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            # apply pre-processing to a RoI extracted from each layer
            rois_ = rois
            roi_feats_t = self.roi_layers[i](feats[i], rois_)
            roi_feats_t = self.pre_conv(roi_feats_t)
            roi_feats_t = self.relu(roi_feats_t)
            # and sum them all
            roi_feats += roi_feats_t

        # apply post-processing before return the result
        x = self.post_conv(roi_feats)
        return x


@ROI_EXTRACTORS.register_module
class ConcatGenericRoiExtractor(SingleRoIExtractor):
    """Extract RoI features from all concatenated feature maps levels.

    Args:
        pre_conf (dict): Specify pre-processing modules.
        reduce_conf (dict): Specify reduce function before post-processing.
        post_conf (dict): Specify post-processing modules.
        ... (see SingleRoIExtractor for other parameters)
    """

    def __init__(self, **kwargs):
        kwargs = deepcopy(kwargs)

        post_conf = kwargs.pop('post_conf')
        pre_conf = kwargs.pop('pre_conf')
        reduce_conf = kwargs.pop('reduce_conf')

        pre_type = pre_conf.pop('type')
        post_type = post_conf.pop('type')

        super(ConcatGenericRoiExtractor, self).__init__(**kwargs)

        # build pre/reduce/post processing modules
        self.post_conv = load_processing_model(post_type, post_conf)
        self.pre_conv = load_processing_model(pre_type, pre_conf)
        self.reduce = ConvModule(**reduce_conf)
        self.relu = nn.ReLU(inplace=False)

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels * num_levels, *out_size)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            # apply pre-processing to a RoI extracted from each layer
            rois_ = rois
            roi_feats_t = self.roi_layers[i](feats[i], rois_)
            roi_feats_t = self.pre_conv(roi_feats_t)
            roi_feats_t = self.relu(roi_feats_t)
            # and concatenates them all
            n_feats = roi_feats_t.shape[1]
            roi_feats[:, i*n_feats:(i+1)*n_feats, ...] = roi_feats_t

        # apply reduce and then post-processing before return the result
        x = self.reduce(roi_feats)
        x = self.relu(x)
        x = self.post_conv(x)
        return x
