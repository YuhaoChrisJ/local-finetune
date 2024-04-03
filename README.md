## Dependencies
Envrionment for finetuning part:

```bash
conda create --name local_ft python==3.8.0
conda activate divcon
```
and install related libraries

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install omegaconf einops kornia
pip install git+https://github.com/CompVis/taming-transformers.git
pip install git+https://github.com/openai/CLIP.git
```

## GLIGEN and SD model weights
Then, download the two models and save them under /model_weights:

```bash
wget https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
```