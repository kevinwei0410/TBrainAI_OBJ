# AI-CUP Competition: STAS Detection

## Environment Setup
Device: single 2080ti with CUDA 10.2 and python3.8
```bash
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
python setup.py install
# python setup.py develop
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

### Apex Installation:
Following [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), we use apex for mixed precision training by default. To install apex, run:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Fix Apex's Bug:
goto your environment folder (eg. venv)   
modify venv/lib/python3.8/site-packages/apex/amp/utils.py line 97
```python 
-   if cached_x.grad_fn.next_functions[1][0].variable is not x:       
+   if cached_x.grad_fn.next_functions[0][0].variable is not x:
        raise RuntimeError("x and cache[x] both require grad, but x is not "
                                   "cache[x]'s parent.  This is likely an error.")
```


## STAS Detection Model and Config
put pretrained pth model in **ckpt** folder (AI-CUP/ckpt)   
>[origin pretrained model](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth.zip)   
>[origin config](https://github.com/VDIGPKU/CBNetV2/blob/main/configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py)     
   
unzip competition models and configs folder and name it **work_dirs** (AI-CUP/work_dirs)  
>[competition models](https://www.dropbox.com/s/xb5g1pyq6fp1vvj/work_dirs.zip?dl=0)    
>[competition segmentation config](https://github.com/jason2714/AI-CUP/blob/main/configs/cbnet/swin_coco.py)    
>[competition detection config](https://github.com/jason2714/AI-CUP/blob/main/configs/cbnet/swin_custom_fine.py)    

## Preprocess STAS Data and Annotations
put data and annotations in **data** folder (AI-CUP/data)    
annotations contain detection and segmentation
```
data
`-- OBJ_Train_Datasets
    |-- Test_Images
    |   `-- test_images...
    |-- Train_Annotations
    |   |-- json_segmentation_annotations...
    |   `-- xml_detection_annotations...
    `-- Train_Images
        `-- train_images...
```

then run **python convert_STAS.py**    
**coco** and **custom** folder should appear in the directory
```
data
`-- OBJ_Train_Datasets
    |-- same_with_above...
    |-- coco
    |   |-- STAS_final.json
    |   |-- STAS_test.json
    |   |-- STAS_train.json
    |   `-- STAS_val.json
    `-- custom
        |-- STAS_final.pkl
        |-- STAS_test.pkl
        |-- STAS_train.pkl
        `-- STAS_val.pkl
```
### COCO Format For Segmentation Annotations (json)

```python
'images': [
    {
        'file_name': '00000395.jpg',
        'height': 942,
        'width': 1716,
        'id': 00000395
    },
    ...
],

'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 00000395,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 0,
        'id': 5555
    },
    ...
],

'categories': [
    {'id': 0, 'name': 'stas'},
 ]
```

### Middle Format For Detection Annotations (pickle)
```python

[
    {
        'filename': '00000395.jpg',
        'width': 1716,
        'height': 942,
        'ann': {
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, )
        }
    },
    ...
]
```

## STAS Detection Train
**Please only use a single GPU for train**     
```bash
# First, train on the semantic segmentation annotations
# The original pretrained model must be placed in the ckpt folder first
python -m torch.distributed.launch tools/train.py 
    configs/cbnet/swin_coco.py
    --gpus 1 --deterministic --seed 123  
    --work-dir work_dirs/swin_coco
    
# Second, use the model trained from segmentation fintune on the object detection annotations
# You need to complete the previous training or download the competition model
python -m torch.distributed.launch tools/train.py 
    configs/cbnet/swin_custom_fine.py 
    --gpus 1 --deterministic --seed 123  
    --work-dir work_dirs/swin_custom_fine
```

## STAS Detection Evaluate
**Please only use a single GPU for inference**    

```bash
# You need to complete all previous training or download the competition model
python tools/test.py 
    work_dirs/swin_custom_fine/swin_custom_fine.py 
    work_dirs/swin_custom_fine/latest.pth 
    --out result.json 
    --show --show-dir ckpt
```

## Other Links
> **Original CBNet**: See [CBNet: A Novel Composite Backbone Network Architecture for Object Detection](https://github.com/VDIGPKU/CBNet).    
> **Origin CBNetV2 Github**: See [VDIGPKU CBNetV2](https://github.com/VDIGPKU/CBNetV2)
## Citation
If you use our code/model, please consider to cite our paper [CBNetV2: A Novel Composite Backbone Network Architecture for Object Detection](http://arxiv.org/abs/2107.00420).
```
@article{liang2021cbnetv2,
  title={CBNetV2: A Composite Backbone Network Architecture for Object Detection}, 
  author={Tingting Liang and Xiaojie Chu and Yudong Liu and Yongtao Wang and Zhi Tang and Wei Chu and Jingdong Chen and Haibing Ling},
  journal={arXiv preprint arXiv:2107.00420},
  year={2021}
}
```
