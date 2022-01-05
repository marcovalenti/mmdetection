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
    CLASSES = ( 'accartocciamento_fogliare', 'black_rot_foglia', 'black_rot_grappolo' ,
		 'botrite_foglia', 'botrite_grappolo', 'carie_bianca_grappolo',
		 'foglia_vite', 'grappolo_vite', 'malattia_esca', 'oidio_foglia',
		 'oidio_grappolo', 'oidio_tralci', 'peronospora_foglia', 'peronospora_grappolo',
		 'red_blotch_foglia', 'virosi_pinot_grigio')