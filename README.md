# DualNet

## Introduction
This is the official PyTorch implementation of the [Artistic Glyph Synthesis via One-Stage Few-Shot Learning]().

- **Architecture**
![Architecture](imgs/architecture.png)

Skip Connection               |  Local Discriminator
:----------------------------:|:-------------------------:
![](imgs/skipconnection.png)  |  ![](imgs/localpatch.png)


## Demo
![](imgs/comparison1.png)

![](imgs/comparison2.png)


## Prerequisites
- Linux or macOS
- CPU or NVIDIA GPU + CUDA cuDNN
- Python 3
- PyTorch 0.4.0+


## Get Started

### Installation
1. Install PyTorch, torchvison and dependencies from [https://pytorch.org](https://pytorch.org)
2. Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate):
   ```shell
   pip install visdom
   pip install dominate
   ```
3. Clone this repo:
   ```shell
   git clone -b master --single-branch https://github.com/hologerry/DualNet
   cd DualNet
   ```

### Datasets
Download the datasets using the following script, four datasets are available.
> It may take a while, please be patient
```
bash ./datasets/download_dataset.sh dataset_name
```
- `base_gray_color` English synthesized gradient glyph dataset, proposed by [MC-GAN](https://arxiv.org/abs/1712.00516).
- `base_gray_texture` English artistic glyph dataset, proposed by [MC-GAN](https://arxiv.org/abs/1712.00516).
- `skeleton_gray_color` Chinese synthesized gradient glyph dataset by us.
- `skeleton_gray_texture` Chinese artistic glyph dataset proposed by us.

### Model Training
- To train a model, download the training images (e.g., English artistic glyph transfer)
  ```shell
  bash ./datasets/download_dataset.sh base_gray_color
  bash ./datasets/download_dataset.sh base_gray_texture
  ```

- Train a model:
  1. Pretrain on synthesized gradient glyph dataset
     ```shell
     bash ./scripts/train.sh base_gray_color GPU_ID
     ```
     > GPU_ID indicates which GPU to use.
  2. Fineture on artistic glyph dataset
     ```shell
     bash ./scripts/train.sh base_gray_texture GPU_ID DATA_ID FEW_SIZE
     ```
     > DATA_ID indicates which artistic font is fine-tuned.  
     > FEW_SIZE indicates the size of few-shot set.  
     
     It will raise an error says
     ```
     FileNodeFoundError: [Error 2] No such file or directory: 'chechpoints/base_gray_texture/base_gray_texture_DATA_ID_TIME/latest_net_G.pth
     ```
     Copy the pretrained model to above path
     ```shell
     cp chechpoints/base_gray_color/base_gray_color_TIME/latest_net_* chechpoints/base_gray_texture/base_gray_texture_DATA_ID_TIME/
     ```
     And start train again. It will works well.

### Model Testing
- To test a model, copy the trained model from `checkpoint` to `pretrained_models` folder (e.g., English artistic glyph transfer)
  ```shell
  cp chechpoints/base_gray_color/base_gray_texture_DATA_ID_TIME/latest_net_* pretrained_models/base_gray_texture_DATA_ID/
  ```

- Test a model
  ```shell
  bash ./scripts/test_base_gray_texture.sh GPU_ID DATA_ID
  ```


## Citation
```
```


## Acknowledgements
This code is inspired by the [BicycleGAN](https://github.com/junyanz/BicycleGAN) repository.