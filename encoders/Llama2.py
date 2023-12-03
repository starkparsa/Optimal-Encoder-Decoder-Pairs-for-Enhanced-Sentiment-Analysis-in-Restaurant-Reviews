import torch
from transformers import AutoTokenizer, AutoModel

class Llama2Embedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        self.model = AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        self.model.eval()  # Set the model to evaluation mode

    def get_embeddings(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs['last_hidden_state'].mean(dim=1).squeeze().numpy()


