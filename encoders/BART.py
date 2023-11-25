import torch
from transformers import BartTokenizer, BartModel

class BartEmbedder:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.model = BartModel.from_pretrained('facebook/bart-large')
        self.model.eval()  # Set the model to evaluation mode

    def get_embeddings(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs['last_hidden_state'].mean(dim=1).squeeze().numpy()