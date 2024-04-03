import torch
from ldm.util import instantiate_from_config


device = "cuda"
saved_ckpt = torch.load("./gligen.bin", map_location=torch.device('cpu'))
config = saved_ckpt["config_dict"]["_content"]
# print(config['model'])
model = instantiate_from_config(config['model']).to(device).eval()
autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
diffusion = instantiate_from_config(config['diffusion']).to(device).eval()
print(model)

# --------------- set opt ---------------#



# --------------- create data ---------------#



# --------------- start training -----------------------#


