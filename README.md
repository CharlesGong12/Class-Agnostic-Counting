## CounTR

#### :boom: :boom: This is our final project for Digital Image Processing, FDU, which achieved an **A grade**!

Thanks to @singularity-s0 and @Dash Kev.

### Project Overview
Our model is based on CounTR. The main changes we implemented are as follows:
1. **Contour-based Counting**: We used OpenCV contour-based counting to assist in density map counting.
2. **Removed Exemplar**: We found that the exemplar interfered with the results in some cases, so we deleted it and turned it into a zero-shot problem.

### Results
- **Validation Set**: 
  - Zero-shot MAE/MSE: **15.90/58.46**
- **Test Set**:
  - Zero-shot MAE/MSE: **13.86/91.51**
  - Contour-based Counting MAE/MSE: **13.81/91.49** (Slightly better than the original CounTR)

### Branches
Our repository includes the following branches, each addressing different aspects and improvements of the CounTR model:

- FSC: Baseline provided by the teaching assistant
- CounTR and CounTR-BackUpVersion: Original CounTR model with modified environment configuration
- Vit-encoder: Baseline using Vit as encoder
- **counting-convnet**: Zero-shot + hybrid counting (**Key improvement branch**)
- countr-clip: Uses CLIP as text encoder for multimodal
- countr-clip-full: Changes both the image and text encoder to CLIP
- **countr-finetune-zs**: Fine-tunes the zero-shot model
- countr-textonly-regression: Based on countr-textonly, uses convolutional network to regress
- **countr-zeroshot**: Zero-shot model (**Key improvement branch**)
- exemplar-resnet: Replaces CounTR's exemplar encoder with pretrained ResNet18
- resnet: Uses residual connections between encoder and decoder


### Here is the original author's Readme：

Details can be found in the paper.

[[Paper]](https://arxiv.org/abs/2208.13721) [[Project page]](https://verg-avesta.github.io/CounTR_Webpage/)

<img src=img/arch.png width="80%"/>

### Contents
* [Preparation](#preparation)
* [CounTR train](#countr-train)
* [CounTR inference](#countr-inference)
* [Fine-tuned weights](#fine-tuned-weights)
* [Visualisation](#visualisation)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

### Preparation
#### 1. Download datasets

In our project, the following datasets are used.
Please visit following links to download datasets:

* [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

* [CARPK](https://lafi.github.io/LPN/)

In fact, we use CARPK by importing hub package. Please click [here](https://docs.activeloop.ai/datasets/carpk-dataset) for more information.

#### 2. Download required python packages:

The following packages are suitable for NVIDIA GeForce RTX 3090.

```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install numpy
pip install matplotlib tqdm 
pip install tensorboard
pip install scipy
pip install imgaug
pip install opencv-python
pip3 install hub
```
* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.


### CounTR Train

Please modify your work directory and dataset directory in the following train files.

|  Task   | model file | train file |
|  ----  | ----  | ----  |
| Pretrain on FSC147 | models_mae_noct.py | FSC_pretrain.py |
| Finetune on FSC147 | models_mae_cross.py | FSC_finetune_cross.py |
| Finetune on CARPK | models_mae_cross.py | FSC_finetune_CARPK.py |

Pretrain on FSC147 

```
CUDA_VISIBLE_DEVICES=0 python FSC_pretrain.py \
    --epochs 500 \
    --warmup_epochs 10 \
    --blr 1.5e-4 --weight_decay 0.05
```

Finetune on FSC147 

```
CUDA_VISIBLE_DEVICES=0 nohup python -u FSC_finetune_cross.py \
    --epochs 1000 \
    --blr 2e-4 --weight_decay 0.05  >>./train.log 2>&1 &
```

Finetune on CARPK

```
CUDA_VISIBLE_DEVICES=0 nohup python -u FSC_finetune_CARPK.py \
    --epochs 1000 \
    --blr 2e-4 --weight_decay 0.05  >>./train.log 2>&1 &
```

### CounTR Inference

Please modify your work directory and dataset directory in the following test files.

|  Task   | model file | test file |
|  ----  | ----  | ----  |
| Test on FSC147 | models_mae_cross.py | FSC_test_cross.py |
| Test on CARPK | models_mae_cross.py | FSC_test_CARPK.py |

Test on FSC147

```
CUDA_VISIBLE_DEVICES=0 nohup python -u FSC_test_cross.py >>./test.log 2>&1 &
```

Test on CARPK

```
CUDA_VISIBLE_DEVICES=0 nohup python -u FSC_test_CARPK.py >>./test.log 2>&1 &
```

Also, demo.py is a small demo used for testing on a single image.

```
CUDA_VISIBLE_DEVICES=0 python demo.py
```

### Fine-tuned weights

benchmark| MAE | RMSE |link|
:---:|:---:|:---:|:---:|
FSC147 | 11.95 (Test set) | 91.23 (Test set) |[weights](https://drive.google.com/file/d/1CzYyiYqLshMdqJ9ZPFJyIzXBa7uFUIYZ/view?usp=sharing) 
CARPK | 5.75 | 7.45 |[weights](https://drive.google.com/file/d/1f0yy4pLAdtR7CL1OzMF123wiHgJ8KpPS/view?usp=sharing)

### Visualisation

<img src=img/goodpred.png width="75%"/>

### Citation

```
@article{liu2022countr,
  author = {Chang, Liu and Yujie, Zhong and Andrew, Zisserman and Weidi, Xie},
  title = {CounTR: Transformer-based Generalised Visual Counting},
  journal = {arXiv:2208.13721},
  year = {2022}
}
```

### Acknowledgements

We borrowed the code from
* [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)
* [MAE](https://github.com/facebookresearch/mae)
* [timm](https://timm.fast.ai/)

Thanks @GioFic95 for adding the function of using external exemplars, more predictions images, more parametrized inference and so on.

If you have any questions about our code implementation, please contact us at liuchang666@sjtu.edu.cn


