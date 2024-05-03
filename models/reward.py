import yaml
import numpy as np
from tqdm import tqdm
import torch

from models.model_itm import ALBEF
from models.vit import interpolate_pos_embed
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
config = yaml.load(open("configs/ITM.yaml", 'r'), Loader=yaml.Loader)
input_resolution = 384
itm_labels = {'negative':0, 'positive':2}
checkpoint_path = "/nfs/turbo/umms-vgvinodv/models/ALBEF/checkpoint_7.pth"

def get_reward_model(): 
    model = ALBEF(config=config, 
                     text_encoder='bert-base-uncased', 
                     tokenizer=tokenizer
                     ).to(device)  
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    state_dict = checkpoint['model']
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    msg = model.load_state_dict(state_dict,strict=False)
    model = model.eval()
    return model