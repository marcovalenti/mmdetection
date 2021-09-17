import glob
import os
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.utils import print_log

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class LeavesDataset(CocoDataset):
    CLASSES = ( 'black_rot_grappolo', 'grappolo_vite', 'foglia_vite' ,
		 'botrite_foglia', 'black_rot_foglia', 'virosi_pinot_grigio', 'red_blotch_foglia',
		 'malattia_esca', 'oidio_tralci', 'carie_bianca_grappolo', 'peronospora_grappolo',
		 'peronospora_foglia', 'oidio_grappolo', 'oidio_foglia', 'botrite_grappolo',
		 'accartocciamento_fogliare')
