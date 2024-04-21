from transformers import BertModel
from transformers import BertTokenizer
from transformers import get_scheduler
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


# class to convert dataset to suitable format 
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokens, tags, pols = self.df.iloc[idx, :3].values

        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')
        pols = pols.strip('][').split(', ')

        bert_tokens = []
        bert_att = []
        pols_label = 0
        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            bert_tokens += t
            if int(pols[i]) != -1:
                bert_att += t
                pols_label = int(pols[i])
        segment_tensor = [0] + [0]*len(bert_tokens) + [0] + [1]*len(bert_att)
        bert_tokens = ['[cls]'] + bert_tokens + ['[sep]'] + bert_att
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        ids_tensor = torch.tensor(bert_ids)
        pols_tensor = torch.tensor(pols_label)
        segment_tensor = torch.tensor(segment_tensor)
        return bert_tokens, ids_tensor, segment_tensor, pols_tensor

    def __len__(self):
        return len(self.df)

# class to get pre-trained BERT model and add him additional layer
class ABSABert(nn.Module):
    def __init__(self, bert_model_name, adapter=True):
        super(ABSABert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.adapter = adapter
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.linear2 = nn.Linear(128, 3)

    def forward(self, ids_tensors, segments_tensors, masks_tensors):
        outputs = self.bert(
            input_ids=ids_tensors,
            token_type_ids=segments_tensors,
            attention_mask=masks_tensors
        )
        pooled_output = outputs[1]
        if self.adapter:
            pooled_output = self.linear1(pooled_output)
            pooled_output = torch.relu(pooled_output)
            pooled_output = self.linear2(pooled_output)
        return pooled_output
        
# class for the generation of the model includes training the new layer and testing the trained model
class ABSAModel():
    def __init__(self, tokenizer, adapter=True):
        self.model = ABSABert('bert-base-uncased', adapter=adapter)
        self.tokenizer = tokenizer
        self.trained = False
        self.adapter = adapter


    def padding(self, samples):
        from torch.nn.utils.rnn import pad_sequence
        ids_tensors = [s[1] for s in samples]
        ids_tensors = pad_sequence(ids_tensors, batch_first=True)

        segments_tensors = [s[2] for s in samples]
        segments_tensors = pad_sequence(segments_tensors, batch_first=True)

        label_ids = torch.stack([s[3] for s in samples])
        
        masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

        return ids_tensors, segments_tensors, masks_tensors, label_ids


    def train_new_layer(self, data, epochs, device, lr=1e-4):
        # Freeze pre-trained layers
        for param in self.model.bert.parameters():
            param.requires_grad = False

        # Define optimizer for the new layer only
        new_layer_params = list(self.model.linear1.parameters()) + list(self.model.linear2.parameters())
        optimizer = torch.optim.AdamW(new_layer_params, lr=lr)
        
        loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=self.padding)
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            for batch in tqdm(loader):
                ids_tensors, segments_tensors, masks_tensors, label_ids = batch
                ids_tensors = ids_tensors.to(device)
                segments_tensors = segments_tensors.to(device)
                label_ids = label_ids.to(device)
                masks_tensors = masks_tensors.to(device)

                optimizer.zero_grad()

                # Forward pass through the new layer only
                outputs = self.model(ids_tensors=ids_tensors, segments_tensors=segments_tensors, masks_tensors=masks_tensors)
                loss = F.cross_entropy(outputs, label_ids)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == label_ids).sum().item()
                total_samples += label_ids.size(0)

            epoch_loss = total_loss / len(loader)
            epoch_accuracy = total_correct / total_samples
            
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")
        self.trained = True

    def _accuracy(self, x, y):
        return torch.mean((x == y).float())
        
    def save_model(self, model_name):
        torch.save(self.model.state_dict(), model_name)

    def load_model(self, model_name):
        self.model.load_state_dict(torch.load(model_name))
        self.trained = True
        
    def test_model(self, data, device):
        loader = DataLoader(data, batch_size=32, shuffle=False, collate_fn=self.padding)
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(loader):
                ids_tensors, segments_tensors, masks_tensors, label_ids = batch
                ids_tensors = ids_tensors.to(device)
                segments_tensors = segments_tensors.to(device)
                label_ids = label_ids.to(device)
                masks_tensors = masks_tensors.to(device)

                outputs = self.model(ids_tensors=ids_tensors, segments_tensors=segments_tensors, masks_tensors=masks_tensors)
                loss = torch.nn.functional.cross_entropy(outputs, label_ids)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == label_ids).sum().item()
                total_samples += label_ids.size(0)

        accuracy = total_correct / total_samples
        average_loss = total_loss / len(loader)

        print(f"Test Accuracy: {accuracy}")
        print(f"Test Loss: {average_loss}")

