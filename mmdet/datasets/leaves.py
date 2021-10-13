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
    CLASSES = ( 'oidio_grappolo', 'grappolo_vite', 'black_rot_foglia' ,
		 'foglia_vite', 'oidio_foglia', 'peronospora_foglia', 'oidio_tralci',
		 'botrite_grappolo', 'accartocciamento_fogliare', 'botrite_foglia',
		 'black_rot_grappolo', 'virosi_pinot_grigio', 'red_blotch_foglia', 'malattia_esca',
		 'carie_bianca_grappolo', 'peronospora_grappolo')
