from ldm.util import instantiate_from_config
import os
from omegaconf import OmegaConf
from transformers import CLIPProcessor, CLIPModel
import torch
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import argparse


device = "cuda"

# --------------- read args ---------------#
parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument("--person_id", type=str, default='1306', help="the person id for finetuning.")
parser.add_argument("--epochs", type=int, default=200, help="the person id for finetuning.")
args = parser.parse_args()


# --------------- load model ---------------#
#read local model
saved_ckpt = torch.load("./model_weights/diffusion_pytorch_model.bin", map_location=torch.device('cpu'))
config = saved_ckpt["config_dict"]["_content"]
config = OmegaConf.create(config)
model = instantiate_from_config(config.model).to(device)
autoencoder = instantiate_from_config(config.autoencoder).to(device)
text_encoder = instantiate_from_config(config.text_encoder).to(device)
diffusion = instantiate_from_config(config.diffusion).to(device)

#load params
model.load_state_dict(saved_ckpt['model'])
autoencoder.load_state_dict(saved_ckpt['autoencoder'])
text_encoder.load_state_dict(saved_ckpt['text_encoder'], strict=False)
diffusion.load_state_dict(saved_ckpt['diffusion'])

def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False
autoencoder.eval()
text_encoder.eval()
disable_grads(autoencoder)
disable_grads(text_encoder)

#grounding_tokenizer_input
grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
model.grounding_tokenizer_input = grounding_tokenizer_input

# --------------- set opt and schduler ---------------#
#load original sd model modules
params = []
trainable_names = []
for name, p in model.named_parameters():
    if ("transformer_blocks" in name) and ("fuser" in name):
        # New added Attention layers
        pass
    elif  "position_net" in name:
        # Grounding token processing network
        pass
    elif (config['inpaint_mode']) and ("input_blocks.0.0.weight" in name):
        # First conv layer in inpaitning model
        pass
    else:
        # Following make sure we do not miss any new params
        # all new added trainable params have to be haddled above
        # otherwise it will trigger the following error
        params.append(p)
        trainable_names.append(name)

opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay)

#set scheduler
def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0
def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=config.warmup_steps)


# --------------- read training data ---------------#
def resize_and_pad(img, output_size, fill_color=(255, 255, 255)):
    original_size = img.size

    ratio = float(output_size) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])

    img = img.resize(new_size, Image.ANTIALIAS)

    new_img = Image.new("RGB", (output_size, output_size), fill_color)

    upper_left = ((output_size - new_size[0]) // 2, (output_size - new_size[1]) // 2)

    new_img.paste(img, upper_left)

    return new_img

def adjust_layout(layout, original_size, output_size):
    ratio = float(output_size) / max(original_size)

    scaled_layout = [int(coord * ratio) for coord in layout]

    new_width = int(original_size[0] * ratio)
    new_height = int(original_size[1] * ratio)
    offset_x = (output_size - new_width) // 2
    offset_y = (output_size - new_height) // 2

    adjusted_layout = [
        scaled_layout[0] + offset_x,
        scaled_layout[1] + offset_y,
        scaled_layout[2] + offset_x,
        scaled_layout[3] + offset_y
    ]

    return adjusted_layout

transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_and_pad(img, 512)),
    transforms.ToTensor(), 
])

img_path = './data/selected_faces/'
info_path = './data/face_data.json'

image_to_load = [f for f in os.listdir(img_path) if f.startswith(args.person_id)]
with open(info_path, 'r') as f:
    info = json.load(f)

images = []
name_layouts = []
for img_name in image_to_load:
  image = Image.open(img_path+img_name).convert("RGB")
  size = image.size
  image = transform(image)
  images.append(image)
  name_layout = info[img_name]
  layout = name_layout[1]
  name_layout[1] = adjust_layout(layout, size, 512)
  name_layouts.append(name_layout)

# --------------- data loader ---------------#
def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature

class MyDataset(Dataset):
    def __init__(self, contexts_and_boxes, images, phrases, transform=None):
        """
        Args:
            texts_and_boxes (list of lists): 每个元素包含文本和对应的box信息的列表。
            images (list of Tensors): 图片tensor列表，每个tensor对应一张图片。
            transform (callable, optional): 可选的转换操作，用于对样本进行处理。
        """
        self.contexts_and_boxes = contexts_and_boxes
        self.images = images
        self.transform = transform
        self.phrases = phrases

    def __len__(self):
        # 假设contexts_and_boxes和images的长度是相等的
        return len(self.contexts_and_boxes)

    def __getitem__(self, idx, max_objs=30):
        """
        根据索引idx获取数据集中的数据，并进行处理。
        """
        phrases = self.phrases
        image = self.images[idx]

        version = "openai/clip-vit-large-patch14"
        model = CLIPModel.from_pretrained(version).cuda()
        processor = CLIPProcessor.from_pretrained(version)

        boxes = torch.zeros(max_objs, 4).to(device)
        masks = torch.zeros(max_objs).to(device)
        text_masks = torch.zeros(max_objs).to(device)
        text_embeddings = torch.zeros(max_objs, 768).to(device)


        phrase_features = get_clip_feature(model, processor, phrases[0], is_image=False)

        text_embeddings[0] = phrase_features
        text_masks[0] = 1 #Here text represents phrase

        context, box = self.contexts_and_boxes[idx]
        box = [coord/512 for coord in box]
        boxes[0] = torch.tensor(box)
        masks[0] = 1
        sample = {'image': image.to(device), 'context': context,
                  "boxes" : boxes,
                  "masks" : masks,
                  "text_masks" : text_masks,
                  "text_embeddings"  : text_embeddings}

        if self.transform:
            sample = self.transform(sample)

        return sample

phrases = ['face']
dataset = MyDataset(name_layouts, images, phrases)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



# --------------- start training -----------------------#

num_train_epochs = args.epochs
epoch_num = 0
for epoch in range(num_train_epochs):
    model.train()
    for batch in dataloader:
        opt.zero_grad()  # 清除梯度
        x = batch['image'].to(device)
        text = batch['context']
        boxes = batch['boxes'].to(device)
        encoded_image = autoencoder.encode(x)
        encoded_texts = text_encoder.encode(text)


        t = torch.randint(0, 1000, (x.size(0),), device=device)  # T是时间步的总数
        noise = torch.randn_like(encoded_image)
        x_noisy = diffusion.q_sample(x_start=encoded_image, t=t, noise=noise)  # 获取对应的噪声水平
        grounding_input = grounding_tokenizer_input.prepare(batch)
        input = dict(x=x_noisy,
                    timesteps=t,
                    context=encoded_texts,
                    inpainting_extra_input=None,
                    grounding_input=grounding_input)
        predict_noise = model(input)
        left, top, right, bottom = boxes[0][0]
        loss = torch.nn.functional.mse_loss(predict_noise[:,:,int(64*top):int(64*bottom),int(64*left):int(64*right)], noise[:,:,int(64*top):int(64*bottom),int(64*left):int(64*right)])
        # loss = torch.nn.functional.mse_loss(predict_noise, noise)

        print(loss)
        loss.backward()
        opt.step()
        scheduler.step()
    epoch_num += 1
    print("epoch ", epoch_num)

# --------------- save as sd architecture -----------------------#
sd_model_path = './model_weights/sd-v1-4.ckpt'
sd_ckpt = torch.load(sd_model_path, map_location="cpu")

i=0
for k,v in sd_ckpt["state_dict"].items():
  if k.startswith('model.diffusion_model'):
    sd_ckpt["state_dict"][k] = params[i]
    i+=1

torch.save(sd_ckpt, "stable-diffusion/models/ldm/stable-diffusion-v1/" + args.person_id + "_tuned_sd.ckpt")