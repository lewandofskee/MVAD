
<div align="center">
<h3>Learning Multi-view Anomaly Detection with Efficient Adaptive Selection</h3>

[Haoyang He<sup>1*</sup>](https://scholar.google.com/citations?hl=zh-CN&user=8NfQv1sAAAAJ),
[Jiangning Zhang<sup>2*</sup>](https://zhangzjn.github.io),
[Guanzhong Tian<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=0q-7PI4AAAAJ),
[Chengjie Wang<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=fqte5H4AAAAJ),
[Lei Xie<sup>1†</sup>](https://scholar.google.com/citations?hl=zh-CN&user=7ZZ_-m0AAAAJ)

<sup>1</sup>College of Control Science and Engineering, Zhejiang University, 
<sup>2</sup>Youtu Lab, Tencent,

[[`Paper`](https://arxiv.org/pdf/2407.11935?)] 

Our MVAD is based on [ADer](https://github.com/zhangzjn/ADer).
</div>

## 📜 Multi-class Results on Real-IAD Multi-View Setting

Subscripts `S`, `I`, and `R` represent `rsample-level`, `image-level`, and `pixel-level`, respectively.

### MVAD Results
|   Dataset    | mAU-ROC<sub>S</sub> | mAP<sub>S</sub> | m*F*1-max<sub>S</sub> | mAU-ROC<sub>I</sub> | mAP<sub>I</sub> | m*F*1-max<sub>I</sub> | mAU-ROC<sub>P</sub> | mAP<sub>P</sub> | m*F*1-max<sub>P</sub> | mAU-PRO<sub>P</sub> |                                                                            <span style="color:blue">Download</span>                                                                            |
|:-----------:|:-------------------:|:---------------:|:-----------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:---------------:|:---------------------:|:-------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Real-IAD   |        90.2         |      95.3       |         90.1          |        86.6         |      84.8       |        77.2          |        97.9         |         30.3          |        36.8         |      91.2       | [log](log/log_test.txt) & [weight](log/mvad.pth) |

## 🛠️ Getting Started

### Installation


- Prepare general experimental environment
  ```shell
  pip3 install timm==0.8.15dev0 mmselfsup pandas transformers openpyxl imgaug numba numpy tensorboard accimage Ninja
  pip3 install --upgrade protobuf==3.20.1 scikit-image faiss-gpu
  pip3 install geomloss FrEIA adeval fvcore==0.1.5.post20221221
  pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
  (or) conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  ```

  
### Dataset Preparation 
### Multi-view Real-IAD Dataset
- Download and extract [Real-IAD](https://realiad4ad.github.io/Real-IAD/) into `data/realiad`.

### Train (Multi-view Anomaly Detection under Multi-class Unsupervised AD Setting)
- Check `data` and `model` settings for the config file `configs/mvad/mvad_realiad.py`
- Train with single GPU example: 
```
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/mvad/mvad_realiad.py -m train
```
- Modify `trainer.resume_dir` to resume training. 


### Test
- The training log of MVAD can be find at [log](log/log_test.txt) and the weights at  [model](log/mvad.pth).
- Modify `trainer.resume_dir` or `model.kwargs['checkpoint_path']`
- Test with single GPU example: 
```
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/mvad/mvad_realiad.py -m test model.model_kwargs.checkpoint_path=log/mvad.pth
```

## Citation
If you find this code useful, don't forget to star the repo and cite the paper:
```
@article{he2024learning,
  title={Learning Multi-view Anomaly Detection},
  author={He, Haoyang and Zhang, Jiangning and Tian, Guanzhong and Wang, Chengjie and Xie, Lei},
  journal={arXiv preprint arXiv:2407.11935},
  year={2024}
}
```