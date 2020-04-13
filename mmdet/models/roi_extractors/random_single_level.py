from __future__ import division

import torch

from ..registry import ROI_EXTRACTORS
from .single_level import SingleRoIExtractor


@ROI_EXTRACTORS.register_module
class RandomSingleRoIExtractor(SingleRoIExtractor):

    def __init__(self, *args, **kwargs):
        super(RandomSingleRoIExtractor, self).__init__(*args, **kwargs)

    def map_roi_levels(self, rois, num_levels):
        return torch.randint(0, num_levels, (len(rois),))
