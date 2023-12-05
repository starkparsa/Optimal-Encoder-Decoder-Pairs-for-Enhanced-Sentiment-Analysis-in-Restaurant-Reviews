import torch
from transformers import BertTokenizer, BertModel

class BertEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def get_embeddings(self, sentence):
        with torch.no_grad():
            tokens = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**tokens)
        return outputs['last_hidden_state'].mean(dim=1).squeeze().numpy()