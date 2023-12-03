import torch
from transformers import T5Model, T5Tokenizer

class T5Embedder:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5Model.from_pretrained('t5-base')
        self.model.eval()  # Set the model to evaluation mode

    def get_embeddings(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs['last_hidden_state'].mean(dim=1).squeeze().numpy()