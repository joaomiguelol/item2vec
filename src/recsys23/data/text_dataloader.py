
import torch
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        # Create a dictionary that maps each item ID to a unique index
        self.vocab = vocab
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # get article_id1 and article_id2 and labels
        article_id1 = row['article_1']
        article_id2 = row['article_2']
        label = row['label']
        # convert to torch tensors
        article_id1 = torch.tensor(self.vocab[article_id1], dtype=torch.long)
        article_id2 = torch.tensor(self.vocab[article_id2], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)

        text_1 = row['text_1']
        text_2 = row['text_2']

        return article_id1, article_id2, text_1, text_2, label