import sys
sys.path.insert(0, './CLIP')   # Add path to local CLIP directory
import clip  #type: ignore
import torch            #type: ignore
import torch.nn as nn               # type: ignore
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torchvision import transforms              # type: ignore
from PIL import Image               # type: ignore

class MappingNetwork(nn.Module):
    def __init__(self, clip_dim=512, gpt_dim=768):
        super(MappingNetwork, self).__init__()
        self.linear = nn.Linear(clip_dim, gpt_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=gpt_dim, nhead=8),
            num_layers=3
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.transformer(x)
        return x

class ClipCap(nn.Module):
    def __init__(self, clip_model, tokenizer, mapping_network):
        super(ClipCap, self).__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.mapping_network = mapping_network
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, images, captions=None):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float()
        prefix = self.mapping_network(image_features)

        if captions is not None:
            embeddings = self.gpt2.transformer.wte(captions)
            embeddings = torch.cat([prefix.unsqueeze(1), embeddings], dim=1)
            outputs = self.gpt2(inputs_embeds=embeddings)
            logits = outputs.logits
            logits = logits[:, 1:, :]
            return logits
        else:
            return self.generate_caption(prefix)

    def generate_caption(self, prefix, max_length=50):
        generated = torch.tensor([self.tokenizer.bos_token_id]).unsqueeze(0).to(prefix.device)
        for _ in range(max_length):
            embeddings = self.gpt2.transformer.wte(generated)
            embeddings = torch.cat([prefix.unsqueeze(1), embeddings], dim=1)
            outputs = self.gpt2(inputs_embeds=embeddings)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

# ---- Set up the model once when the backend starts ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_model, preprocess_clip = clip.load('ViT-B/32', device=device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

mapping_network = MappingNetwork().to(device)
clipcap_model = ClipCap(clip_model, tokenizer, mapping_network).to(device)

# Load your trained weights
checkpoint_path = 'model/clipcap_final_model.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
clipcap_model.load_state_dict(checkpoint['model_state_dict'])
clipcap_model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

# ---- Function to generate caption ----
def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = clipcap_model(image)

    return caption