import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

import pycocotools.mask as mask_util
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def get_gt_masks(dataset, reference_labels):
    '''
    Lo scopo di questo metodo è ottenere per ogni immagine del test set di ground truth
    una lista di dict pari al numero di immagini, dove ogni elemento è la percentuale di malattia
    rispetto alla superclasse di riferimento
    '''
    classes = dataset.CLASSES
    masks_result = {}
    result = []
    
    for i in range(len(dataset)):
        #ottengo l'info della immagine i-esima
        #restituita come dict
        img_info = dataset.get_ann_info(i)
        labels = img_info['labels']
        masks = img_info['masks']
        
        for l in range(len(labels)):
            bin_mask = mask_util.decode(masks[l])
            
            if labels[l] in masks_result.keys():
                temp = masks_result[labels[l]].copy()
                masks_result[labels[l]] = np.bitwise_or(bin_mask, temp)
                
            else:
                masks_result[labels[l]] = bin_mask
    
        result.append(masks_result)
        masks_result = {}
            
    for i in range(len(result)):
        for key in result[i].keys():
            #sostituisco alle mappe binarie la loro area occupata
            tmp = np.copy(result[i][key])
            result[i][key] = np.sum(tmp)
        
        for key in result[i].keys():
            area_temp = result[i][key].copy()
            
            if 'grappolo' in classes[key] and classes[key] != 'grappolo_vite':
                #NOTE qua non si fa alcun controllo sull'esistenza o meno della
                #classe base per il calcolo della percentuale
                result[i][key] = (area_temp / result[i][reference_labels['grappolo_vite']])
            
            elif 'foglia' in classes[key] and classes[key] != 'foglia_vite':
                #NOTE qua non si fa alcun controllo sull'esistenza o meno della
                #classe base per il calcolo della percentuale
                result[i][key] = (area_temp / result[i][reference_labels['foglia_vite']])
                
            elif classes[key] == 'virosi_pinot_grigio' or classes[key] == 'malattia_esca':
                result[i][key] = (area_temp / result[i][reference_labels['foglia_vite']])
                
        
        if reference_labels['foglia_vite'] in result[i].keys():
            result[i].pop(reference_labels['foglia_vite'])
        
        if reference_labels['grappolo_vite'] in result[i].keys():
            result[i].pop(reference_labels['grappolo_vite'])
        
                          
    return result
       
               
def get_test_masks(dataset, outputs, reference_labels, show_thr=0.3):
    '''
    Lo scopo di questo metodo è ottenere per ogni immagine del test set ottenuto dalla rete
    una lista di dict pari al numero di immagini, dove ogni elemento è la percentuale di malattia
    rispetto alla superclasse di riferimento
    '''
    classes = dataset.CLASSES
    class_result = {}
    result = []
    
    for id_img in range(len(outputs)):
        #outputs[id_img][0] -> bbox
        #outputs[id_img][0] -> segm
        for id_classe in range(len(outputs[id_img][1])):
            
            if len(outputs[id_img][1][id_classe]) > 0:
                #considero le classi che hanno una segmentazione
                h, w = outputs[id_img][1][id_classe][0]['size']
                res_mask = np.array([0] * h * w, dtype = 'uint8')
                res_mask = res_mask.reshape((h, w))
                empty_mask = res_mask.copy()
                
                for id_segm in range(len(outputs[id_img][1][id_classe])):
                    #considero solo le segmentazioni sopra soglia
                    net_confidence = outputs[id_img][0][id_classe][id_segm][4]
                    
                    if net_confidence < show_thr:
                        res_mask = np.bitwise_or(res_mask, empty_mask)
                    else:
                        #decodifico la maschera e ci faccio l'or con la maschera
                        #finale della classe
                        temp = res_mask.copy()
                        bin_mask = outputs[id_img][1][id_classe][id_segm]
                        res_mask = np.bitwise_or(temp, mask_util.decode(bin_mask))
                    
                class_result[id_classe] = res_mask
                
        result.append(class_result)
        class_result = {}

        for key in result[id_img].keys():
            #sostituisco alle mappe binarie la loro area occupata
            tmp = np.copy(result[id_img][key])
            result[id_img][key] = np.sum(tmp)
        
        for key in result[id_img].keys():
            area_temp = result[id_img][key].copy()
            
            if 'grappolo' in classes[key] and classes[key] != 'grappolo_vite' and reference_labels['grappolo_vite'] in result[id_img].keys():
                if result[id_img][reference_labels['grappolo_vite']] > 0:
                    result[id_img][key] = (area_temp / result[id_img][reference_labels['grappolo_vite']])
                else:
                    result[id_img][key] = 0.0
                
            elif 'grappolo' in classes[key] and classes[key] != 'grappolo_vite' and reference_labels['grappolo_vite'] not in result[id_img].keys():
                #NOTE: se esiste la malattia ma non la classe base, metto a 0 la percentuale di malattia
                result[id_img][key] = 0.0
            
            elif 'foglia' in classes[key] and classes[key] != 'foglia_vite' and reference_labels['foglia_vite'] in result[id_img].keys():
                if result[id_img][reference_labels['foglia_vite']] > 0:
                    result[id_img][key] = (area_temp / result[id_img][reference_labels['foglia_vite']])
                else:
                    result[id_img][key] = 0.0
                
            elif 'foglia' in classes[key] and classes[key] != 'foglia_vite' and reference_labels['foglia_vite'] not in result[id_img].keys():
                #NOTE: se esiste la malattia ma non la classe base, metto a 0 la percentuale di malattia
                result[id_img][key] = 0.0
                
            elif classes[key] == 'virosi_pinot_grigio' or classes[key] == 'malattia_esca' and reference_labels['foglia_vite'] in result[id_img].keys():
                if result[id_img][reference_labels['foglia_vite']] > 0:
                    result[id_img][key] = (area_temp / result[id_img][reference_labels['foglia_vite']])
                else:
                    result[id_img][key] = 0.0
                
            elif classes[key] == 'virosi_pinot_grigio' or classes[key] == 'malattia_esca' and reference_labels['foglia_vite'] not in result[id_img].keys():
                #NOTE: se esiste la malattia ma non la classe base, metto a 0 la percentuale di malattia
                result[id_img][key] = 0.0
                 
        
        if reference_labels['foglia_vite'] in result[id_img].keys():
            result[id_img].pop(reference_labels['foglia_vite'])
        
        if reference_labels['grappolo_vite'] in result[id_img].keys():
            result[id_img].pop(reference_labels['grappolo_vite'])
        
    return result
    
def compute_disease_area_error(ground_truth_res, evaluation_results):
    '''
    lo scopo di questo metodo è calcolare l'errore medio della
    percentuale di malattia stimata dalla rete, rispetto
    alla percentuale di ground truth
    '''
    error = []
    class_error = {}
    for id_img in range(len(ground_truth_res)):
        for key in ground_truth_res[id_img].keys():
            if key in evaluation_results[id_img].keys():
                class_error[key] = abs(ground_truth_res[id_img][key] - evaluation_results[id_img][key])
            else:
                class_error[key] = ground_truth_res[id_img][key]
            
        for eval_key in evaluation_results[id_img].keys():
            if eval_key not in ground_truth_res[id_img].keys() and evaluation_results[id_img][eval_key] > 0.0:
                class_error[eval_key] = evaluation_results[id_img][eval_key]
        
        error.append(class_error)
        class_error = {}
    
    return error         
             
def compute_avg_error_classwise(area_error, classes):
    count = [0] * len(classes)
    class_error = [0.0] * len(classes)
    
    for img_err in area_error:
        for key in img_err.keys():
            count[key] += 1
            class_error[key] += img_err[key]
    
    for i in range(len(class_error)):
        if count[i] > 0:
            temp = class_error[i]
            class_error[i] = temp / count[i]
        else:
            class_error[i] = 0.0
    
    avg_error = {}
    for i in range(len(class_error)):
        avg_error[classes[i]] = class_error[i]
        
    return avg_error

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

    '''dopo aver ottenuto i risultati, effettuo
    il calolo della percentuale della malattia rispetto 
    alla superclasse rispettiva(grappolo o foglia) sia per il
    ground truth che per le maschere ottenute dal test e le confronto
    per calcolare l'errore per immagine di area malata e poi estrarre
    l'errore medio per classe'''
    
    reference_labels = {}
    for j in range(len(dataset.CLASSES)):
        if dataset.CLASSES[j] == 'foglia_vite':
            reference_labels['foglia_vite'] = j
        elif dataset.CLASSES[j] == 'grappolo_vite':
            reference_labels['grappolo_vite'] = j
        elif dataset.CLASSES[j] == 'oidio_tralci':
            reference_labels['oidio_tralci'] = j
    
    gt_results = get_gt_masks(dataset, reference_labels)
    eval_results = get_test_masks(dataset, outputs, reference_labels, args.show_score_thr)
    area_error = compute_disease_area_error(gt_results, eval_results)
    
    avg_error = compute_avg_error_classwise(area_error, dataset.CLASSES)
    avg_error.pop('oidio_tralci')
    avg_error.pop('grappolo_vite')
    avg_error.pop('foglia_vite')
    print(avg_error)

if __name__ == '__main__':
    main()
