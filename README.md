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