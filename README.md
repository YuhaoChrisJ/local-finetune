## Dependencies
Envrionment for finetuning part:

```bash
conda create --name local_ft python==3.8.0
conda activate local_ft
```
and install related libraries, make sure your pytorch is associated with cuda version

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install omegaconf einops kornia transformers Pillow==8.0.0 pytorch_lightning
pip install git+https://github.com/openai/CLIP.git
```

## GLIGEN and SD model weights
Then, download the two models and save them under /model_weights:

```bash
wget -P model_weights https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin
wget -P model_weights https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
```