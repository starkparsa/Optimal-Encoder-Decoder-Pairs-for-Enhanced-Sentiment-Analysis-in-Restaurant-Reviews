import torch
from transformers import AutoTokenizer, AutoModel

class ErnieEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
        self.model =  AutoModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
        self.model.eval()  # Set the model to evaluation mode

    def get_embeddings(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs['last_hidden_state'].mean(dim=1).squeeze().numpy()