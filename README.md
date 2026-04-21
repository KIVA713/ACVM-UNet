# ACVM-UNet
ACVM-UNet:Adaptive Content-Aware Vision Mamba for Medical Image Segmentation

We proposes ACVM-UNet, an improved network for medical image segmentation. Building upon the VM-UNet framework, the proposed method introduces the RCARAFE (Residual Content-Aware ReAssembly of FEatures) upsampling module and the AMVSSBLOCK (Adaptive Multi-scale Visual State Space Block) for multi-scale feature modeling. RCARAFE enhances local neighborhood information interaction through a content-aware reassembly mechanism and incorporates a residual interpolation branch to restore spatial continuity, effectively addressing boundary blurring and structural fragmentation. Meanwhile, AMVSSBLOCK strengthens feature representation across different receptive fields via parallel multi-scale convolutions and an adaptive weight fusion mechanism, thereby improving the model's generalization capability.

<img width="1115" height="1142" alt="image" src="https://github.com/user-attachments/assets/259296cd-2cb7-4f3c-b761-b8787e2f2c28" />

# Main Environments
<pre>
conda create -n acvmunet python=3.7
conda activate acvmunet
matplotlib  
SimpleITK 
scipy
scikit-learn
scikit-image 
pillow 
torch 
torchvision
torchaudio
wheel
yacs 
torch 
mmcv-full
opencv-python 
causal-conv1d
timm
pytest 
chardet
termcolor
packaging
  </pre>

# Train the VM-UNet
<pre>
cd ACVM-UNet
python train.py
</pre>
# Citation
<pre>
If you find our work helpful, please consider citing:
Meanwhile, please cite the baseline work, VM-UNet:
```latex
@article{ruan2024vmunet,
  title={VM-UNet: Vision Mamba UNet for Medical Image Segmentation},
  author={Ruan, Jiacheng and Xiang, Suncheng and others},
  journal={arXiv preprint arXiv:2402.02491},
  year={2024}
}
This project is licensed under the Apache License 2.0. Parts of the code are derived from VM-UNet and are used under its original license.
</pre>
