# ContextMRI
This repository is the official implementation of "ContextMRI: Enhancing Compressed Sensing MRI through Metadata Conditioning"

---
## 🔥 Summary

**VideoGuide** 🚀 enhances temporal quality in video diffusion models *without additional training or fine-tuning* by leveraging a pretrained model as a guide. During inference, it uses a guiding model to provide a temporally consistent sample, which is interpolated with the sampling model's output to improve consistency. VideoGuide shows the following advantages:

1. **Improved temporal consistency** with preserved imaging quality and motion smoothness
2. **Fast inference** as application only to early steps is proved sufficient
4. **Prior distillation** of the guiding model

## 🗓 ️News
- [9 Jan 2025] Code are uploaded.

## 🛠️ Setup
First, create your environment. We recommend using the following comments. 

```
git clone https://github.com/DoHunLee1/ContextMRI.git
cd ContextMRI

conda create -n contextmri python=3.10
conda activate contextmri
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## ⏳ Models

|Models|Checkpoints|
|:---------|:--------|
|ContextMRI|[Hugging Face](https://huggingface.co/DHCAI/ContextMRI)

## 🌄 Example
An example of using **ContextMRI** is provided in the inference.sh, recon_mri.sh code. Also you can train using train.sh for Text-conditioned MRI foundation model.

## 📝 Citation
If you find our method useful, please cite as below or leave a star to this repository.

## 🤗 Acknowledgements
We thank the contributors of [DeepFloyd](https://github.com/deep-floyd/IF) for sharing their awesome work. 

> [!note]
> This work is currently in the preprint stage, and there may be some changes to the code.
