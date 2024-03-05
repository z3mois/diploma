import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
from torch import Tensor
from .utils_local import cosine_similarity


class SentenceBertTransformer:
    def __init__(self, model_name='setu4993/LaBSE', device='cpu', max_len=512):
        self.model_name = model_name
        self.device = device
        self.max_len = max_len
        
    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def average_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def transform(self, text):
        if 'e5' in self.model_name:
            text = 'query: ' + text
        inputs = self.tokenizer(
            [text], return_tensors="pt", padding=True, max_length=self.max_len, verbose=False, truncation=True
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if 'e5' in self.model_name:
            embeddings = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            return embeddings[0].detach().cpu().numpy()
        return outputs[1][0].detach().cpu().numpy()
    
    def transform_batch(self, texts, bs=32):
        batch_count = len(texts) // bs + int(len(texts) % bs != 0)
        results = []
        for i in tqdm(range(batch_count)):
            batch = texts[i * bs: (i + 1) * bs]
            inputs = self.tokenizer(
                batch, return_tensors='pt', padding=True, max_length=self.max_len, verbose=False, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results.append(outputs[1].detach().cpu().numpy())

        torch.cuda.empty_cache()
        return np.vstack(results)
    
    def cosine_similarity(self, sent1:str, sent2:str) -> float:
        '''
        Calculating the cosine proximity between two sentences 
            Param: two sentences
            return: cosine similarity range, -1 to 1
        '''
        sent1_embeding = self.transform(sent1)
        sent2_embeding = self.transform(sent2)
        return cosine_similarity(sent1_embeding, sent2_embeding)
