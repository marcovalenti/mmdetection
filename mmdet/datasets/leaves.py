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
    CLASSES = ( 'black_rot_grappolo', 'grappolo_vite', 'virosi_pinot_grigio',
		 'foglia_vite', 'red_blotch_foglia', 'malattia_esca',
 		 'carie_bianca_grappolo', 'peronospora_grappolo', 'peronospora_foglia',
		 'oidio_tralci', 'oidio_grappolo', 'oidio_foglia', 'botrite_grappolo',
		 'botrite_foglia', 'black_rot_foglia', 'accartocciamento_fogliare')
