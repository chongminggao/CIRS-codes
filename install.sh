#!/bin/bash

#conda create --name CIRS python=3.9

# Attention: Using higher torch version is at your discretion. For example, torch==1.10.0+cu113 and torch==1.10.0+cu111 may result an error:
# RuntimeError: cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED, when calling `cusolverDnXgeqrf( handle, params, m, n, CUDA_R_32F, reinterpret_cast<void*>(A), lda, CUDA_R_32F, reinterpret_cast<void*>(tau), CUDA_R_32F, reinterpret_cast<void*>(bufferOnDevice), workspaceInBytesOnDevice, reinterpret_cast<void*>(bufferOnHost), workspaceInBytesOnHost, info)`
#pip3 install torch==1.9.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html # stable version when writing this repository

pip install torch

pip3 install --upgrade --no-cache-dir scipy sklearn tqdm logzero gym seaborn openpyxl paramiko jupyterlab

cd tianshou
pip3 install -e .
cd ..

cd DeepCTR-Torch
pip3 install -e .
cd ..

cd environments/VirtualTaobao
pip3 install -e .
cd ../..

