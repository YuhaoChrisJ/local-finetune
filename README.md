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

## INFERENCE
Start fine tuning with the following code. The finetuned model will be saved under "stable-diffusion/models/ldm/stable-diffusion-v1/". 

```bash
python local fine_finetune.py --person_id 1306 --epochs 200
```


## INFERENCE
Now navigate to stable-diffusion, follow the instruction to create a new envrionment, then do the inference with your tuned ckpt file.

```bash
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 
```
