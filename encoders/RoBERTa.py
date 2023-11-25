import torch
from transformers import BertTokenizer, BertModel

class RobertaEmbedder:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.model.eval()  # Set the model to evaluation mode

    def get_embeddings(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs['last_hidden_state'].mean(dim=1).squeeze().numpy()