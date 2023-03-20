<p align="center">
  <div align="center">
    <h1>Deep Probabilistic Feature-metric Tracking</h1>
  </div>
  <p align="center">
    <a href="https://binbin-xu.github.io/"><strong>Binbin Xu</strong></a>
    ·
    <a href="https://www.doc.ic.ac.uk/~ajd/"><strong>Andrew J. Davison</strong></a>
    ·
    <a href="https://mlr.in.tum.de/members/leuteneg"><strong>Stefan Leutenegger</strong></a>
  </p>
  <!-- <h2 align="center">In Review</h2> -->
  <h3 align="center">
    <a href="https://arxiv.org/pdf/2008.13504.pdf">Paper</a> |
    <a href="https://youtu.be/6pMosl6ZAPE">Video</a> |
  </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="#">
    <img src="https://binbin-xu.github.io/pub/ral2020/ral2020.gif" alt="">
  </a>
</p>

## Summary

This is the official repository of our RA-L 2021 paper:

**Deep Probabilistic Feature-metric Tracking**, \
*Binbin Xu, Andrew J. Davison, Stefan Leutenegger*, \
IEEE Robotics and Automation Letters (RA-L), Vol. 6, No. 1, pp. 223-230, 2021 (ICRA 2021 presentation) \
Best Paper Honorable Mention Award \
[[Paper]](https://arxiv.org/pdf/2008.13504.pdf) [[Video]](https://youtu.be/6pMosl6ZAPE)


## Setup 
You can reproduce the setup by using our anaconda environment configurations. We have provided an Makefile to help you install the environment. 
``` bash!
make install
```

Everytime before you run, activate the environment inside the repo folder

``` bash!
source .anaconda3/bin/activate
```
The pre-trained network weights can be downloaded at [here](https://imperialcollegelondon.box.com/s/xryhbshxtktizjw5fpmxaic1kncxr4cw).

## Prepare the datasets 

**TUM RGBD Dataset**: Download the dataset from [TUM RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset/download) to '$YOUR_TUM_RGBD_DIR'. Create a symbolic link to the data directory as 

```
ln -s $YOUR_TUM_RGBD_DIR code/data/data_tum
```

**MovingObjects3D Dataset** Download the dataset from [MovingObjs3D](https://drive.google.com/open?id=1EIlS4J2J0sdsq8Mw_03DXHlRQmfL8XQx) to '$YOUR_MOV_OBJS_3D_DIR'. Create a symbolic link to the data directory as 

```
ln -s $YOUR_MOV_OBJS_3D_DIR code/data/data_objs3D
```

**Custom Dataset** You can also use your own dataset. 
Our work use the above two datasets for training and deployed the trained weights on scannet and our self-collected dataset. Please refer to the [ScanNet](code/data/ScanNet.py) and [VaryLighting.py](code/data/VaryLighting.py) for the custom dataloading.


## Training and Evaluation
To run the full training and evaluation, please follow the steps below.

### Run training

**Train example with TUM RGBD dataset:** 

``` bash! 
./scripts/train_tum_rgbd.sh
```

To check the full training setting, run the help config as 
``` bash!
python train.py --help
``` 


**Train example with MovingObjects3D:** Camera egocentric motion is dfifferent from the object-centric motion estimation and thus we provide a separate training script for the MovingObjects3D dataset.
All the same as the last one only except changing the dataset name. You can also use our provided script to train the model. 

``` bash!
./scripts/train_moving_objs3d.sh
```

### Run evaluation
**Run the pretrained model:** If you have set up the dataset properly with the datasets, you can run the learned model with the checkpoint we provided in the trained model directory.

``` bash!
./scripts/eval_tum_rgbd.sh
```


You can substitute the trajectory, the keyframe and the checkpoint file. The training and evaluation share the same config setting. To check the full setting, run the help config as

``` bash!
python evaluate.py --help
```

**Results:** The evaluation results will be generated automatically in both '.pkl' and '*.csv' in the folder 'test_results/'.


**Run comparisons:** We also provide the scripts to run the comparisons with the classic RGBD and ICP methods from Open3D. Please refer to [rgbd_odometry.py](code/tools/rgbd_odometry.py) and [ICP.py](code/tools/ICP.py) for the details accordingly. 


### Joint feature-metric and geometric tracking 
We can combine our proposed feature-metric tracking with the geometric tracking methods to achieve better performance. We provide the scripts to run the joint tracking with the ICP methods.
``` bash!
./scripts/train_tum_feature_icp.sh
```
It is achieved by using the trained feature-metric network weights as the initialization and combing with the ICP methods as the refinement. 

The evaluation script is also provided as
```
./scripts/eval_tum_feature_icp.sh
```


### Run visual odometry
Please note this is a prototype version of our **visual odometry frontend**. It mainly serves as a demo to show the performance of our method. 

``` bash!
./scripts/run_kf_vo.sh
```

To visualise the keyframe tracking in the paper, add the argument `--vo_type keyframe --two_view` to the above script.
To check the full setting, run the help config as 
``` bash!
python code/experiments/kf_vo.py --help
``` 


**Convergence basin analysis** for the keyframe visual odometry is also provided. Check the script `scripts/run_kf_vo_cb.sh` for more details.







## Citation
```bibtex
@article{Xu:etal:RAL2021,
 author = {Binbin Xu and Andrew Davison and Stefan Leutenegger},
 journal = {{IEEE} Robotics and Automation Letters ({RAL})},
 title = {Deep Probabilistic Feature-metric Tracking},
  year={2021},
  volume = {6},
  number = {1},
  pages = {223 - 230},
}
```

Please cite the paper if you found our provided code useful for you. 


## License
This repo is BSD 3-Clause Licensed. Part of its code is from [Taking a Deeper Look at the Inverse Compositional Algorithm](https://github.com/lvzhaoyang/DeeperInverseCompositionalAlgorithm), which is MIT licensed. We thank the authors for their great work.

Copyright © 2020-2021 Smart Robotics Lab, Imperial College London \
Copyright © 2020-2021 Binbin Xu 


## Contact
Binbin Xu (b.xu17@imperial.ac.uk)
