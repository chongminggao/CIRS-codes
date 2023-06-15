# Counterfactual Interactive Recommender System (CIRS)

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/chongminggao/CIRS-codes/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9.0+cu111-%237732a8)](https://pytorch.org/)

This repository contains the official Pytorch implementation for the paper *CIRS: Bursting Filter Bubbles by Counterfactual Interactive Recommender System*. It also contains the two environments in the paper: *VirtualTaobao* (with the leave mechanism altered to penalize filter bubbles) and the proposed *KuaishouEnv*.

<img src="figs/intro2.png" alt="introduction" style="zoom:100%;" />

More descriptions are available via the [paper](https://arxiv.org/pdf/2204.01266.pdf) and the [slides](https://cdn.chongminggao.top/files/pdf/CIRS-slides.pdf).

If this work helps you, please kindly cite our papers:

```latex
@article{gao2022cirs,
  title = {CIRS: Bursting Filter Bubbles by Counterfactual Interactive Recommender System},
  author = {Gao, Chongming and Lei, Wenqiang and Chen, Jiawei and Wang, Shiqi and He, Xiangnan and Li, Shijun and Li, Biao and Zhang, Yuan and Jiang, Peng},
  journal={arXiv preprint arXiv:2204.01266},
  year={2022}
}

@article{gao2022kuairec,
  title={KuaiRec: A Fully-observed Dataset for Recommender Systems},
  author={Gao, Chongming and Li, Shijun and Lei, Wenqiang and Li, Biao and Jiang, Peng and Chen, Jiawei and He, Xiangnan and Mao, Jiaxin and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2202.10842},
  year={2022}
}
```

---
## Two environments for evaluating filter bubbles in interactive recommendation.

- #### VirtualTaobao

The details of *VirtualTaobao* can be referred to [this repository](https://github.com/eyounx/VirtualTaobao). Note that we alter the exit mechanism to penalize filter bubbles. Specifically, in the original VirtualTaobao environment the length of interaction trajectory is fixed and predicted in advance, we change it so that the interaction will be terminated when the recommended items repeat in a short time. 

**Exiting mechanism**: We compute the Euclidean distance between the recommended target and the most recent $N$ recommended items. If any of them is lower than the threshold $d_Q$, the environment will quit the interaction process as the real users can get bored and quit given the tedious recommendation. 



- #### KuaishouEnv

*KuaishouEnv* is created by us in this project to evaluate interactive recommenders in video recommendation on Kuaishou, a video-sharing mobile App. Unlike VirtualTaobao which simulates real users by training a model on Taobao data, we use real user historical feedback in our environment. 

It contains two matrices: *big matrix* and *small matrix*, where the latter is a fully filled user-item matrix. The statistics are shown in the following table. The details of data collection can be referred to the KuaiRec dataset ([Webpage](https://chongminggao.github.io/KuaiRec/), [Paper](https://arxiv.org/pdf/2202.10842.pdf)). 

<img src="figs/KuaiRec.png" alt="KuaishouEnv" style="zoom: 60%;" />


**Exiting mechanism**: For the most recent $N$ recommended items, if more than the threshold $n_Q$ items have at least one attribute of the currently recommended target, then the environment ends the interaction process.

<img src="figs/exit.png" alt="exit" style="zoom:67%;" />

---
## Installation

1. Clone this git repository and change directory to this repository:

	```bash
	git clone git@github.com:chongminggao/CIRS-codes.git
	cd CIRS-codes
	```


2. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested. 

    ```bash
    conda create --name CIRS python=3.9 -y
    ```

3. Activate the newly created environment.

    ```bash
    conda activate CIRS
    ```


4. Install the required 

    ```bash
    sh install.sh
    ```

Note that the implementation requires two platforms, [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) and [Tianshou](https://github.com/thu-ml/tianshou). The codes of the two platforms have already been included in this repository and are altered here and there. 

## Download the data

1. Download the compressed dataset

    ```bash 
    wget https://linux.chongminggao.top/CIRS/environment%20data.zip
    ```

(Download options: If the download via `wget` is too slow, you can manually download the file to the root path of this repository)

- Optional link 1: [Google drive](https://drive.google.com/file/d/1v9y-nxhrtOg_Kd3sm6hJ4curNFpgRbPx/view). 

- Optional link 2: [USTC drive](https://rec.ustc.edu.cn/share/0fcb0130-5bce-11ec-be8a-9b5319b7bbe2)

2. Uncompress the downloaded `environment data.zip` and put the files in their corresponding positions.

   ```bash
   unzip "environment data.zip"
   mv environment\ data/KuaishouEnv/* environments/KuaishouRec/data/
   mv environment\ data/VirtualTaobao/* environments/VirtualTaobao/virtualTB/SupervisedLearning/
   ```
   

If things go well, you can run the following examples now.

---
## Examples to run the code

The following commands only give one argument `--cuda 0` as an example. For more arguments, please kindly refer to the paper and the definitions in the code files. 

- #### VirtualTaobao

1. Train the user model on historical logs

    ```bash
    python3 CIRS-UserModel-taobao.py --cuda 0
    ```

2. Plan the RL policy using a trained user model

    ```bash
    python3 CIRS-RL-taobao.py --cuda 0 --epoch 100 --message "my-CIRS"
    ```

---

- #### KuaishouEnv

1. Train the user model on historical logs

    ```bash
    python3 CIRS-UserModel-kuaishou.py --cuda 0
    ```

2. Plan the RL policy using trained user model

    ```bash
    python3 CIRS-RL-kuaishou.py --cuda 0 --epoch 100 --message "my-CIRS"
    ```

---
## A guide to reproduce the main results of the paper.

You can follow the guide to reproduce the main results of our paper, see [this link](./reproduce_results_of_our_paper).



## Main Contributors

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
        ​ <a href="https://github.com/chongminggao"><img width="70" height="70" src="https://github.com/chongminggao.png?s=40" alt="pic"></a><br>
        ​ <a href="https://github.com/chongminggao">Chongming Gao</a> ​
        <p>
        USTC <br> (中科大)  </p>​
      </td>
      <td>
         <a href="https://github.com/Strawberry47"><img width="70" height="70" src="https://github.com/Strawberry47.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/Strawberry47">Shiqi Wang</a> ​
        <p>Chongqing University <br> (重庆大学)  </p>​
      </td>
    </tr>
  </tbody>
</table>
