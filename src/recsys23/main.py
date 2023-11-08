import config
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertTokenizerFast


train = pd.read_parquet(os.path.join(config.data_processed_dir , 'train_pairs.parquet'))
valid = pd.read_parquet(os.path.join(config.data_processed_dir , 'valid_pairs.parquet'))

articles = pd.read_parquet(os.path.join(config.data_raw_dir , 'articles.parquet'))
articles = articles[['article_id', 'prod_name', 'detail_desc']]

# join prod_name and prod_desc
articles['text'] = articles['prod_name'] + ' ' + articles['detail_desc']
articles['text'] = articles['text'].str.lower()
articles = articles[['article_id', 'text']]
articles['text'] = articles['text'].fillna('') 
# print mac of words in text


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
max_len = 32 + 16
articles['text'] = articles['text'].apply(lambda x: tokenizer(x, truncation=True, padding='max_length', max_length=max_len,return_tensors="pt")['input_ids'][0])


def merge_articles(df, articles, article_column, text_column):
    df = df.merge(articles, left_on=article_column, right_on='article_id', how='left')
    df = df.rename(columns={'text': text_column})
    return df.drop(columns=['article_id'])

# Use the helper function to merge train and valid with article_1 and article_2
train = merge_articles(train, articles, 'article_1', 'text_1')
train = merge_articles(train, articles, 'article_2', 'text_2')

valid = merge_articles(valid, articles, 'article_1', 'text_1')
valid = merge_articles(valid, articles, 'article_2', 'text_2')

train.drop(columns=['customer_id'], inplace=True)
valid.drop(columns=['customer_id'], inplace=True)
train.dropna(inplace=True)
valid.dropna(inplace=True)

print(valid.label.mean())
print(train.label.mean())

df = pd.concat([train, valid], ignore_index=True)
item_ids = set(df['article_1'].unique()).union(set(df['article_2'].unique()))
# Create a dictionary that maps each item ID to a unique index
vocab = {item_id: i for i, item_id in enumerate(item_ids)}
num_items = len(set(df['article_1'].unique()).union(set(df['article_2'].unique())))
del item_ids, df




class PairDataset(Dataset):
    def __init__(self, df, vocab):
        articles = set(df.article_1.unique())
        articles = np.random.permutation(list(articles))
        
        grouped = df.groupby('article_1')

        # Shuffle the groups
        shuffled_df = pd.concat(
            [grouped.get_group(group) for group in articles],
            ignore_index=True
        )
        self.df = shuffled_df
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
    # add the shuffle method
from torch.utils.data import Sampler

class GroupedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.groups = self.data_source.df.groupby('article_1').indices

    def __iter__(self):
        # Shuffle the groups
        group_order = list(self.groups.keys())
        np.random.shuffle(group_order)

        # Yield samples in the order defined by the shuffled groups
        for group in group_order:
            for idx in self.groups[group]:
                yield idx

    def __len__(self):
        return len(self.data_source)
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
dataset = PairDataset(train, vocab)
sampler = GroupedSampler(dataset)
data_loader = DataLoader(dataset, batch_size=256, sampler=sampler)

valid_dataset = PairDataset(valid, vocab)
sampler_valid = GroupedSampler(valid_dataset)
valid_data_loader = DataLoader(valid_dataset, batch_size=1000,  sampler=sampler_valid)




def loss_function(output, target):
    # use cross entropy loss
    return F.binary_cross_entropy(output, target)


def train_model(model, data_loader, optimizer, num_epochs):
    print(device)
    model = model.to(device)

    for epoch in range(num_epochs):
        # switch model to training mode
        model.train()
        with tqdm(total=len(data_loader)) as progress_bar:
            total_loss = 0
            total_accuracy = 0
            for item1, item2, text1, text2, target in data_loader:
                optimizer.zero_grad()
                
                output1 = model(item1.to(device), text1.to(device)) #torch.Size([4000, 128])
                output2 = model(item2.to(device), text2.to(device)) #torch.Size([4000, 128])
                

                dot_product = torch.sum(output1 * output2, dim=1)

                # Apply sigmoid to convert the dot product to a similarity score
                similarity_score = torch.sigmoid(dot_product)
                target = target.float().to(device)
                loss = loss_function(similarity_score, target)

                loss.backward()
                optimizer.step()
                

                
                accuracy = ((similarity_score > 0.5) == target).detach().cpu().numpy().mean()
                total_accuracy += accuracy
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item() , accuracy=accuracy.item()    )
                progress_bar.update(1)
            print('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, total_loss / len(data_loader), total_accuracy / len(data_loader)))
                
                

        # compute total loss and accuracy
        total_loss = 0
        total_accuracy = 0

        # switch model to evaluation mode
        model.eval()

        with torch.no_grad():
            with tqdm(total=len(valid_data_loader)) as progress_bar:
                
                for item1, item2, text1, text2, target in valid_data_loader: 
                    
                    output1 = model(item1.to(device), text1.to(device))
                    output2 = model(item2.to(device), text2.to(device))


                    dot_product = torch.sum(output1 * output2, dim=1)

                    # Apply sigmoid to convert the dot product to a similarity score
                    output = torch.sigmoid(dot_product)
                    
                    loss = loss_function(output, target.float().to(device))


                    # compute accuracy
                    output = output.detach().cpu().numpy()
                    target = target.detach().cpu().numpy()
                    accuracy = ((output > 0.5) == target).mean()
                    total_loss += loss.item()
                    total_accuracy += accuracy
                    progress_bar.update(1)

        print('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, total_loss / len(valid_data_loader), total_accuracy / len(valid_data_loader)))
        # save model
        
    torch.save(model.state_dict(), os.path.join(config.models_dir, 'model.pth'))
        
print('num_tokens: ', len(tokenizer))

model = 'lstm'
if model == 'lstm':
    from models.lstm import Item2Vec
    model = Item2Vec(num_items=num_items, embedding_dim=128, input_tokens=len(tokenizer))
else :
    from models.transformer import Item2Vec
    model = Item2Vec(num_items=num_items, embedding_dim=256, input_tokens=len(tokenizer), max_len=2000)

train_model(model, data_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001), num_epochs=20)