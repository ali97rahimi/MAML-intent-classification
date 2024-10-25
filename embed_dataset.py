import torch.utils.data as data
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/LaBSE')
# model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
# model = SentenceTransformer("HooshvareLab/bert-base-parsbert-uncased")

class EmbedDataset(data.Dataset):
    def __init__(self, root, filename):
        csv_file_path = root + '/' + filename
        self.df = pd.read_csv(csv_file_path)
        self.texts = self.df['Text'].tolist()
        self.intents = self.df['Intent'].tolist()
        
    def __getitem__(self, index):
        text = self.texts[index]
        intent = self.intents[index]
        
        # LaBSE or BERT embedding
        embeddings = model.encode(text)

        return embeddings, intent
        
    def __len__(self):
        return len(self.df)