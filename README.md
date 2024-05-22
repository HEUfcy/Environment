# Vim-Light

### 整体环境
```
git clone https://github.com/HEUfcy/Vim-Light.git
cd Vim-Light

conda create -n visionmamba python=3.10
conda activate visionmamba
conda install -c nvidia cuda-nvcc=11.8.89 # nvcc版本在11.8，CUDA版本应高于11.8
pip3 install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118 # 版本很重要
```

### 先安装causal_sonvld-1.1.3
```
pip install causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# 检查是否安装成功
import torch
import causal_conv1d_cuda
```
  
### 再安装mamba_ssm-1.1.1
```
pip install mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# 检查是否安装成功
import torch
import mamba_ssm
```

### 检查环境
```
python Vim.py
# 运行得到：
# preds shape if torch.Size([4, 1000])
