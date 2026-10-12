# DualViewMapDet - Leveraging Previous-Traversal Point Cloud Map Priors for Camera-Based 3D Object Detection and Tracking

[**arXiv**](https://arxiv.org/abs/2604.25405) | [**Website**](http://dualviewmapdet.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/0p6te677lXU)

This repository is the official implementation of the paper:

> **Leveraging Previous-Traversal Point Cloud Map Priors for
Camera-Based 3D Object Detection and Tracking**
>
> [Markus Käppeler](https://rl.uni-freiburg.de/people/kaeppelm), Özgün Cicek, Yakov Miron, and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada). <br>
>
>

<p align="center">
  <img src="resources/overview.png" alt="Overview of dualviewmapdet approach" width="800" />
</p>

If you find our work useful in your research or applications, please consider giving us a star or citing our paper:
```
@article{kaeppeler2026dvmd,
      author={Käppeler, Markus and Çiçek, Özgün and Miron, Yakov and Valada, Abhinav},
      title={Leveraging Previous-Traversal Point Cloud Map Priors for Camera-Based 3D Object Detection and Tracking}, 
      journal={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2026},
}
```

## News
* **`28 September, 2026`:** We released the code of DualViewMapDet.
* **`17 June, 2026`:** DualViewMapDet was accepted at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026. We will release the code soon. Stay tuned!

## 📔 Abstract
Camera-based 3D object detection and tracking are central to autonomous driving, yet precise 3D object localization remains fundamentally constrained by depth ambiguity when no expensive, depth-rich online LiDAR is available at inference. In many deployments, however, vehicles repeatedly traverse the same environments, making static point cloud maps from prior traversals a practical source of geometric priors. We propose DualViewMapDet, a camera-only inference framework that retrieves such map priors online and leverages them to mitigate the absence of a LiDAR sensor during deployment. The key idea is a dual-space camera-map fusion strategy that avoids one-sided view conversion. Specifically, we (i) project the map into perspective view (PV) and encode multi-channel geometric cues to enrich image features and support BEV lifting, and (ii) encode the map directly in bird's-eye view (BEV) with a sparse voxel backbone and fuse it with lifted camera features in a shared metric space. Extensive evaluations on nuScenes and Argoverse 2 demonstrate consistent improvements over strong camera-only baselines, with particularly strong gains in object localization. Ablations further validate the contributions of PV/BEV fusion and prior-map coverage.

## 📊 Results

## Argoverse 2 Benchmark

### Results on Validation Split

|      Model      |   Backbone    |                                        Pretrain                                        | Image Size | Training Range | Evaluation Range |  CDS  |  mAP  | config |                                           ckpt                                           | Notes                                                                                   |
|:---------------:|:-------------:|:--------------------------------------------------------------------------------------:|:----------:|:--------------:|:----------------:|:-----:|:-----:|:------:|:----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------|
| DualViewMapDet  |   VoVNet-99   | [FCOS3D](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth) | 640x960    |      150m      |       150m       | 0.202 | 0.263 |  [cfg](projects/configs/dualviewmapdet_argoverse2_150m_vov99_fcos_960x640.py) |                                [dualviewmapdet_argoverse2_150m_vov99_fcos_960x640.pth](https://github.com/robot-learning-freiburg/dualviewmapdet/releases/download/v1.0/dualviewmapdet_argoverse2_150m_vov99_fcos_960x640.pth)                              |                                                                                         |
| DualViewMapDet |   VoVNet-99   | [FCOS3D](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth) |  640x960   |      150m      |       50m        | 0.311 | 0.397 |  [cfg](projects/configs/dualviewmapdet_argoverse2_150m_vov99_fcos_960x640.py) |               [dualviewmapdet_argoverse2_150m_vov99_fcos_960x640.pth](https://github.com/robot-learning-freiburg/dualviewmapdet/releases/download/v1.0/dualviewmapdet_argoverse2_150m_vov99_fcos_960x640.pth)               | Set eval_range_m=50.0 in data_basic_config of the config file to obtain the 50m results. |
| DualViewMapDet |   VoVNet-99   | [FCOS3D](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth) |  256x704   |      50m       |         50m      |      0.309       | 0.401 |  [cfg](projects/configs/dualviewmapdet_argoverse2_50m_vov99_fcos_704x480_ablationmain.py) |                            [dualviewmapdet_argoverse2_50m_vov99_fcos_704x480_ablationmain.pth](https://github.com/robot-learning-freiburg/dualviewmapdet/releases/download/v1.0/dualviewmapdet_argoverse2_50m_vov99_fcos_704x480_ablationmain.pth)                            |                                                                                          |


## nuScenes Benchmark
### Results on Validation Split

|    Model    | Backbone |                                                                                         Pretrain                                                                                          | Image Size |  NDS  |  mAP  | AMOTA | AMOTP | IDS | config |            ckpt            |
|:-----------:|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------:|:-----:|:-----:|:-----:|:-----:|:---:| :---:  |:--------------------------:|
| DualViewMapDet | ResNet50 | [ImageNet](https://download.pytorch.org/models/resnet50-19c8e357.pth) |  256x704   | 0.588 | 0.481 | 0.513 | 1.109 | 473 | [cfg](projects/configs/dualviewmapdet_nuscenes_resnet50_nuimg_704x256.py) | [dualviewmapdet_nuscenes_resnet50_nuimg_704x256](https://github.com/robot-learning-freiburg/dualviewmapdet/releases/download/v1.0/dualviewmapdet_nuscenes_resnet50_nuimg_704x256.pth) |

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication [**Leveraging Previous-Traversal Point Cloud Map Priors for Camera-Based 3D Object Detection and Tracking**](https://arxiv.org/abs/2604.25405) . It will neither be maintained nor monitored in any way.

## 👩‍💻 Code

### 🏗 Setup

#### ⚙️ Installation

##### Create a new conda environment
```bash
conda create -n dualviewmapdet python=3.10 -y
conda activate dualviewmapdet
```

##### Install dependencies
```bash
dualviewmapdet_path="path/to/dualviewmapdet"
cd ${dualviewmapdet_path}
# PyTorch
conda install -y   "defaults::mkl<2024"   "defaults::intel-openmp<2024"   -c pytorch -c nvidia -c defaults # fix mkl error when importing torch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia/label/cuda-11.7.0
pip install numpy==1.24.0
conda install xformers==0.0.18 -c xformers
# CUDA 11.7
conda install -y \
  nvidia/label/cuda-11.7.0::cuda-toolkit \
  nvidia/label/cuda-11.7.0::cuda-cccl \
  nvidia/label/cuda-11.7.0::cuda-cudart-dev \
  nvidia/label/cuda-11.7.0::cuda-nvcc \
  nvidia/label/cuda-11.7.0::cuda-libraries-dev
  # GNU C and C++ Compiler
conda install -y gcc_linux-64=11.2.0 gxx_linux-64=11.2.0
pip install "setuptools<70" wheel packaging
pip install -U openmim==0.3.9
mim install mmcv-full==1.7.1
pip install -r requirement.txt
```
##### Install dependencies for Argoverse 2
```bash
pip install av2==0.2.1
pip install refile==0.4.1
```
##### Compile CUDA ops
```bash
# Compile the deformable attention CUDA op
cd projects/mmdet3d_plugin/ops
cd ms_deformable_attention/
python setup.py build install 
cd ..
# Compile the deformable aggregation CUDA op
python3 setup.py develop
cd ../../../
```
##### Fix error in tracking evaluation lib
```bash
# Fix the error in tracking evaluation lib (due to the change of Iterable in Python 3.10)
nano /home/username/miniconda3/envs/dualviewmapdet/lib/python3.10/site-packages/motmetrics/metrics.py
Change "from collections import Iterable" to "from collections.abc import Iterable"
```
##### KISS-Matcher to refine GT poses of Argoverse 2
```bash
cd ..
git clone git@github.com:MIT-SPARK/KISS-Matcher.git
cd KISS-Matcher/python
conda install -c conda-forge flann=1.9.* eigen ninja pkg-config
export CXXFLAGS="-include cassert ${CXXFLAGS}"
pip3 install --no-build-isolation -e . -v   --config-settings=cmake.build-type=Release   --config-settings=cmake.args="-DCMAKE_POLICY_VERSION_COMPATIBLE=3.5;-DCMAKE_POLICY_VERSION_MINIMUM=3.5"  # Build KISS-Matcher from source
```

#### 📁 Dataset

##### Prepare the nuScenes data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.
```bash
cd ${dualviewmapdet_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos.
```bash
sh scripts/create_data.sh
```

##### Prepare the Argoverse 2 data
Download the [Argoverse 2 dataset](https://www.argoverse.org/av2.html#download-link) and put Argoverse 2 in /path/to/av2, create symbolic links.
```bash
cd ${dualviewmapdet_path}
mkdir data
ln -s path/to/av2 ./data/av2
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos.
```bash
export PYTHONPATH="$(pwd)":$PYTHONPATH
python tools/data_converter/av2_gather_argo2_anno_feather.py
python tools/data_converter/av2_converter.py
```

##### Generate anchors by K-means
Generated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
python tools/kmeans/kmeans_det.py
python tools/kmeans/kmeans_det_argoverse2_50m.py # Argoverse 2 50m
python tools/kmeans/kmeans_det_argoverse2_150m.py # Argoverse 2 150m
```

##### Refine ground-truth LiDAR poses on Argoverse 2 to compute correct poses
We refine the global ego poses since the given 6 DoF global ego poses are not always aligned between sequences. There can be a translation offset of up to a few meters between sequences. Within a sequence, the ego poses are correct.
Compute the refined global ego poses your own with our script below or download them here: [lidarpose.zip](https://github.com/robot-learning-freiburg/dualviewmapdet/releases/download/v1.0/lidarpose.zip)
```bash
python ./tools/global_map/av2_lidarpose_refine.py 2>&1 | tee logs/av2_lidar_pose_refine.txt
```

##### Generate global LiDAR point cloud map
Argoverse 2:
```bash
./tools/dist_train.sh projects/configs/global_map_generator_av2_lidar_config.py 1 --deterministic --no-validate 2>&1 | tee logs/global_map_generator_av2_lidar_config_train.txt
./tools/dist_train.sh projects/configs/global_map_generator_av2_lidar_config_val.py 1 --deterministic --no-validate 2>&1 | tee logs/global_map_generator_av2_lidar_config_val.txt
```

nuScenes:
```bash
./tools/dist_train.sh projects/configs/global_map_generator_nuscenes_lidar_config.py 1 --deterministic --no-validate 2>&1 | tee logs/global_map_generator_nuscenes_lidar_config_train.txt
./tools/dist_train.sh projects/configs/global_map_generator_nuscenes_lidar_config_val.py 1 --deterministic --no-validate 2>&1 | tee logs/global_map_generator_nuscenes_lidar_config_val.txt
```

Copy a subset of Argoverse 2 data to a local server disk to accelerate training. (optional, can be used if training with load_interval>=1)
```bash
# Train
python tools/utils/copy_av2_subset.py \
  --data-root data/av2 \
  --pkl data/av2_ssd/av2_train_infos.pkl \
  --split train \
  --load-interval 5 \
  --dst usename@server:/some_ssd/av2/ \
#  --dry-run

# Val
python tools/utils/copy_av2_subset.py \
  --data-root data/av2 \
  --pkl data/av2_ssd/av2_val_infos.pkl \
  --split val \
  --load-interval 5 \
  --dst username@server:/some_ssd/av2/ \
#  --dry-run
```


### 🏋️‍♂️ Training

##### Download pre-trained weights
Download the required backbone.
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O ckpt/resnet18-5c106cde.pth
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
wget https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth  -O ckpt/fcos3d_vovnet_imgbackbone-remapped.pth
```

#### Training
Start training:
```bash
nohup ./tools/dist_train.sh projects/configs/some_config_file.py 8 --deterministic > logs/some_config_file.txt 2>&1 &
```
Resume training:
```bash
nohup ./tools/dist_train.sh projects/configs/some_config_file.py 8 --deterministic --resume-from ./work_dirs/some_config_file/latest.pth > logs/some_config_file.txt 2>&1 &
```
Use tmux instead of nohup to keep the training running in the background. (recommended)
```bash
tmux new -s session_name # New session
./tools/dist_train.sh projects/configs/some_config_file.py 8 --deterministic 2>&1 | tee logs/some_config_file.txt # start run
./tools/dist_train.sh projects/configs/some_config_file.py 8 --deterministic --resume-from ./work_dirs/some_config_file/latest.pth 2>&1 | tee logs/some_config_file.txt # resume run
tmux attach -t session_name # Attach to session
```

#### Training on SLURM GPU cluster
```bash
./tools/slurm_train.sh job_name projects/configs/some_config_file.py
```

### 📈 Evaluation
```bash
./tools/dist_test.sh projects/configs/some_config_file.py ckpt/some_config_file.pth 8 --deterministic --eval bbox # --result_file ./work_dirs/some_config_file/results.pkl
```

Evaluate on subsets of sequences with higher prior-map coverage (Table IV. in paper):
```bash
# Find AV2 sequences (scene_token) whose entire tile coverage is overlapped by other sequences.
# To reproduce overlap k=1, set min_other_sequences_per_tile = 1 in the script below. For overlap k = 2, set min_other_sequences_per_tile = 2.
# We also provide these .txt files in ./data/av2_gmaps/overlap_scene_tokens_minother1.txt (overlap k=1) and ./data/av2_gmaps/overlap_scene_tokens_minother2.txt (overlap k=2)
python -u ./tools/global_map/av2_determine_overlapping_seq.py 2>&1 | tee logs/av2_determine_overlapping_seq.txt
# To evaluate on this subset, set "keep_scene_tokens_txt" to the generated .txt file in the config file (data["test"]["keep_scene_tokens_txt"] = ""./data/av2_gmaps/overlap_scene_tokens.txt")
./tools/dist_test.sh projects/configs/some_config_file.py ckpt/some_config_file.pth 8 --deterministic --eval bbox # --result_file ./work_dirs/some_config_file/results.pkl
```

#### Visualization
```bash
export PYTHONPATH="$(pwd)":$PYTHONPATH
./tools/dist_test.sh projects/configs/some_config_file.py work_dirs/some_config_file/epoch_[number].pth 8 --deterministic --out work_dirs/some_config_file/results_epoch_[number].pkl
python tools/visualization/visualize.py \
	projects/configs/some_config_file.py \
	--result-path work_dirs/some_config_file/results.pkl
```

### Useful commands
Kill training run:
```bash
kill -SIGINT $(ps aux | grep tools/train.py | grep -v grep | awk '{print $2}')
```

List running GPU processes:
```bash
ps -up `nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//'`
```

## ⚖️ License
DualViewMapDet is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE.txt) file for details.

For a list of other open source components included in DualViewMapDet, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## 🙏 Acknowledgment

This research was funded by Bosch Research as part of a collaboration between Bosch Research and the University of Freiburg on AI-based automated driving. 

- [SparseDrive](https://github.com/swc-17/SparseDrive)
- [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)
- [BEVNeXt](https://github.com/woxihuanjiangguo/BEVNeXt)
- [SparseBEV](https://github.com/MCG-NJU/SparseBEV)
- [Far3D](https://github.com/megvii-research/Far3D)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [LT3D](https://github.com/neeharperi/LT3D)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)


