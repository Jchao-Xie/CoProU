<div align="center">

# CoProU-VO: Combining Projected Uncertainty for End-to-End Unsupervised Monocular Visual Odometry

### **[Jingchao Xie\*](https://www.linkedin.com/in/jingchao-xie-16b724297)\***<sup>1,3</sup>, **[Oussema Dhaouadi\*](https://cvg.cit.tum.de/members/dhou)\***<sup>1,2,3</sup>†, **[Weirong Chen](https://wrchen530.github.io/)**<sup>1,3</sup>, **[Johannes Meier](https://cvg.cit.tum.de/members/mejo)**<sup>1,2,3</sup>, **[Jacques Kaiser](https://jacqueskaiser.com/)**<sup>2</sup>, **[Daniel Cremers](https://cvg.cit.tum.de/members/cremers)**<sup>1,3</sup>

<sup>1</sup> [Computer Vision Group at Technical University of Munich (TUM)](https://cvg.cit.tum.de/)  
<sup>2</sup> [DeepScenario](https://www.deepscenario.com)  
<sup>3</sup> [Munich Center for Machine Learning (MCML)](https://mcml.ai)

\* Shared first authorship  † Corresponding author  

</div>


## Contribution

We present **Combined Projected Uncertainty Visual Odometry (CoProU-VO)** —  
a novel visual odometry approach that robustly handles regions violating the static scene assumption within an unsupervised visual odometry framework.

![Uncertainty Visualization](assets/image.png)

**Figure**: Gray areas in the images indicate invalid regions excluded from loss calculation. Photometric residual brightness represents error magnitude, while projection brightness reflects uncertainty.  **Dynamic objects** may appear distorted due to the static scene assumption.  Our method robustly masks high-uncertainty regions, distinguishes parked cars (e.g., green box) from moving cars (e.g., red boxes), and detects **occluded parts** of parked vehicles (e.g., yellow box).  

## Preparation

### Environment

```bash
conda create -n coprou python=3.9
conda activate coprou

# Install PyTorch and torchaudio (version 2.7.0 with CUDA 11.8 support)
# ⚠️ Make sure to install the version that matches your local CUDA version.
# You can find other compatible versions at https://pytorch.org/get-started/previous-versions/
pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# We use xFormers==0.0.30. Make sure to install a version compatible with your installed PyTorch version.
pip install xformers==0.0.30 --extra-index-url https://download.pytorch.org/whl/cu118

# Install other required Python packages
pip install -r requirements.txt
```

### Datasets and Preprocessing

We trained and evaluated our model on two datasets:

- **[KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)**  

- **[nuScenes Dataset](https://www.nuscenes.org/nuscenes#download)**  

Please download the datasets from the official links above and organize them under the `\storage` directory as follows:

```bash
\storage
  \KITTI_odometry
    \dataset
      \sequences
        ...
  \nuScenes
    \maps
    \samples
    \sweeps
    \v1.0-trainval
    ...
```
Please Use the following commands to preprocess the datasets.
#### KITTI Odometry
```bash
python data/prepare_train_data.py storage/KITTI_odometry/dataset \
    --dataset-format 'kitti_odom' \
    --dump-root storage/kitti_vo_256/ \
    --width 832 --height 256 \
    --num-threads 4
```

#### nuScenes
```bash
python data/nusc.py --config data/nuscenes_config/local_nusc.yaml
```
Processed data will be saved under folder `\storage`
### Checkpoints
Create folder `\checkpoints`,
```bash
mkdir -p checkpoints
```
and put the following checkpoints under the created folder.
#### CoProU-VO
Download **[CoProU-VO checkpoints](https://drive.google.com/drive/folders/1_wW9djPj9zWqflDkW6NsF11I5MIu5G1S?usp=drive_link)**.

#### Pre-trained ViTs 

Download **[Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true)** and **[ViT-S/14 distilled](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)**

```bash
# Download Depth-Anything-V2-Small checkpoint
wget -O checkpoints/depth_anything_v2_vits.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true"

# Download ViT-S/14 distilled (DINOv2) checkpoint
wget -O checkpoints/dinov2_vits14_pretrain.pth "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"

```


## Inference and Visualization

Once the dataset and checkpoint are prepared, inference on two consecutive images can be performed using the following command as an example:


### KITTI
```bash
python intermediate_visualization.py \
  --pretrained-dispnet checkpoints/dispnet_checkpoint_kitti.pth.tar \
  --pretrained-posenet checkpoints/exp_pose_checkpoint_kitti.pth.tar \
  --img-height 256 \
  --img-width 832 \
  --dataset kitti\
  --tgt-img storage/kitti_vo_256/05_2/002354.jpg \
  --ref-img storage/kitti_vo_256/05_2/002353.jpg

```

### nuScenes
```bash
python intermediate_visualization.py \
  --pretrained-dispnet checkpoints/dispnet_checkpoint_nusc.pth.tar \
  --pretrained-posenet checkpoints/exp_pose_checkpoint_nusc.pth.tar \
  --img-height 256 \
  --img-width 416 \
  --dataset nuscenes\
  --tgt-img storage/nuscenes_416_256/scene-0685_0/n008-2018-08-28-16-16-48-0400__CAM_FRONT__1535488216112404.jpg\
  --ref-img storage/nuscenes_416_256/scene-0685_0/n008-2018-08-28-16-16-48-0400__CAM_FRONT__1535488216262404.jpg

```

Outputs, including depths, uncertainties, and synthesized image will be saved under `\visualization`.


## Training

### Training on KITTI:

```bash
torchrun --nproc_per_node=2 --master-port=29755 lightning_train.py storage/kitti_vo_256 --dataset kitti \
--encoder vits --dan \
--epochs 75 -b12 -s0.1 -c0.6 --sequence-length 3 \
--with-ssim 1 --with-mask 0 --with-auto-mask 1 --with-pretrain 1 \
--name kitti --lr 5e-4 
```

### Training on nuScenes

```bash
torchrun --nproc_per_node=4 --master-port=29755 lightning_train.py storage/nuscenes_416_256 --dataset nuscenes \
--encoder vits --dan \
--epochs 25 -b8 -s0.1 -c0.6 --skip-frames 2 --sequence-length 3 \
--with-ssim 1 --with-mask 0 --with-auto-mask 1 --with-pretrain 1 \
--name nusc --lr 5e-4 
```

#### Tensorboard and Checkpoints are saved under `\checkpoints`. 


## Evaluation

### Evaluation of our provided checkpoints

On KITTI:
```bash
python test_vo.py --pretrained-posenet checkpoints/exp_pose_checkpoint_kitti.pth.tar --img-height 256 --img-width 832 --dataset-dir storage/KITTI_odometry/dataset/sequences/ --sequence 09 --output-dir eval_result/kitti/

python kitti_eval/eval_odom.py --result=eval_result/kitti/ --align='7dof'
```

On nuScenes:
```bash
# eval
python test_vo_nusc.py --pretrained-posenet checkpoints/exp_pose_checkpoint_nusc.pth.tar --img-height 256 --img-width 416 --dataset-dir storage/nuscenes_416_256/ --output-dir eval_result/nusc

python nusc_eval/eval_odom.py --result=eval_result/nusc/checkpoints/exp_pose_checkpoint_nusc/ --align='7dof'
```

```bash
# test
python test_vo_nusc.py --test --pretrained-posenet checkpoints/exp_pose_checkpoint_nusc.pth.tar --img-height 256 --img-width 416 --dataset-dir storage/nuscenes_416_256/ --output-dir eval_result/nusc

python nusc_eval/eval_odom.py --test --result=eval_result/nusc/checkpoints/exp_pose_checkpoint_nusc/ --align='7dof'
```


### Evaluation of your trained checkpoints

On KITTI:
```bash
python test_vo.py --pretrained-model <path to the checkpoints auto-saved by training script> --img-height 256 --img-width 832 --dataset-dir storage/KITTI_odometry/dataset/sequences/ --sequence 09 --output-dir eval_result/kitti/

python kitti_eval/eval_odom.py --result=eval_result/kitti/ --align='7dof'
```

On nuScenes:
```bash
# eval
python test_vo_nusc.py --pretrained-model <path to the checkpoints auto-saved by training script>  --img-height 256 --img-width 416 --dataset-dir storage/nuscenes_416_256/ --output-dir eval_result/nusc

python nusc_eval/eval_odom.py --result=eval_result/nusc/checkpoints/+'<name of your checkpoint>' --align='7dof'
```

```bash
# test
python test_vo_nusc.py --test --pretrained-model <path to the checkpoints auto-saved by training script>  --img-height 256 --img-width 416 --dataset-dir storage/nuscenes_416_256/ --output-dir eval_result/nusc

python nusc_eval/eval_odom.py --test --result=eval_result/nusc/checkpoints/+'<name of your checkpoint>' --align='7dof'
```

## If you find our work useful in your research please consider citing our paper:
 
    @article{xie2025gcpr, 
      title={{CoProU-VO:} Combining Projected Uncertainty for End-to-End Unsupervised Monocular Visual Odometry}, 
      author={Xie, Jingchao and Dhaouadi, Oussema and Chen, Weirong and Meier, Johannes and Kaiser, Jacques and Cremers, Daniel}, 
      journal= {DAGM German Conference on Pattern Recognition}, 
      year={2025} 
    }

## Visual Odometry Results on KITTI odometry dataset 

#### CoProU-VO result trained on sequence 00, 02-08

|Metric               | Seq. 09 | Seq. 10 |
|---------------------|---------|---------|
|ATE (m)              | 9.84    |11.28    |
|t_err (%)            | 4.56    | 7.76    |
|r_err (degree/100m)  | 2.02    | 3.58    | 



    
 ##  Acknowledgement 
 
We appreciate the contributions of the following projects, which have greatly supported our work:

 * [SfMLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) - A pioneering framework for end-to-end monocular visual odometry.

 * [SC-Depth](https://github.com/JiawangBian/sc_depth_pl) - Our baseline.
 
 * [Kitti-Odom-Eval-Python](https://github.com/Huangying-Zhan/kitti-odom-eval) - Python implementation for KITTI odometry evaluation.
 
 * [RoGS](https://github.com/fzhiheng/RoGS) - Preprocessing code for the nuScenes dataset.

 
 * [DepthAnything-v2](https://github.com/DepthAnything/Depth-Anything-V2) and [DINOv2](https://github.com/facebookresearch/dinov2) – Providing Vision Transformer backbone features.

 ## License

This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](./LICENSE) file for more details.
 
