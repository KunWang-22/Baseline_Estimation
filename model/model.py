from ast import arg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import os
import argparse
import random

seed = 22
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



parser = argparse.ArgumentParser("The hyper-parameters of this project")

# model parameters
parser.add_argument('--num_layer', type=int, default=3)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--d_in', type=int, default=1)
parser.add_argument('--d_out', type=int, default=1)
parser.add_argument('--d_hidden', type=int, default=32)
parser.add_argument('--d_key', type=int, default=4)
parser.add_argument('--d_value', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.3)

# training parameters
parser.add_argument('--file_path', type=str, default="../data/full_data.csv")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--num_critic', type=int, default=3)
parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
parser.add_argument('--element_loss_weight', type=float, default=2.0)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--beta_1', type=float, default=0.9)
parser.add_argument('--beta_2', type=float, default=0.999)
parser.add_argument('--cuda', action='store_true', default=False)

args = parser.parse_args()

args.cuda = not args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super(PositionwiseFeedForward, self).__init__()
        self._linear_1 = nn.Linear(d_model, d_hidden)
        self._linear_2 = nn.Linear(d_hidden, d_model)
        self._activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear_2(self._activation(self._linear_1(x)))
        return x

class AddNorm(nn.Module):
    def __init__(self, d_model: int):
        super(AddNorm, self).__init__()
        self._layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        x = self._layer_norm(x+residual)
        return x

class EmbeddingPositionEncoding(nn.Module):
    def __init__(self, d_in: int, d_model: int):
        super(EmbeddingPositionEncoding, self).__init__()
        self._d_model = d_model
        self._linear = nn.Linear(d_in, d_model)

    def forward(self, x: torch.Tensor):
        length = x.shape[1]
        x = self._linear(x)
        x = x + self._position_encoding(length, self._d_model).to(device)
        return x

    def _position_encoding(self, length: int, d_model: int):
        encoding = torch.zeros((length, d_model))
        position = torch.arange(length).unsqueeze(1)
        encoding[:, 0::2] = torch.sin( position / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model) )
        encoding[:, 1::2] = torch.cos( position / torch.pow(10000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model) )
        return encoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, heads: int):
        super(MultiHeadAttention, self).__init__()
        self._heads = heads
        # d_q is equal to d_k, we use d_k here
        self._W_q = nn.Linear(d_model, self._heads*d_k)
        self._W_k = nn.Linear(d_model, self._heads*d_k)
        self._W_v = nn.Linear(d_model, self._heads*d_v)
        self._W_o = nn.Linear(self._heads*d_v, d_model)
        self._softmax = nn.Softmax(dim=-1)
        # Placeholder
        self._attention_scores = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: str = None):
        sequence_length = key.shape[1]
        # For parallel computing, split the featrure dimension and concat into batch dimension. Very ingenious !!!
        # The new batch dimensions is add after the original dimensions. 
        queries = torch.cat( self._W_q(query).chunk(self._heads, dim=-1), dim=0 )
        keys = torch.cat( self._W_q(key).chunk(self._heads, dim=-1), dim=0 )
        values = torch.cat( self._W_q(value).chunk(self._heads, dim=-1), dim=0 )

        self._attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(sequence_length)

        # For masked attention in decoder
        if mask == 'decoder':
            masks = (torch.triu( torch.ones((sequence_length, sequence_length)), diagonal=1 ).bool()).to(self._attention_scores.device)
            self._attention_scores = self._attention_scores.masked_fill(masks, float('-inf'))

        self._attention_scores = self._softmax(self._attention_scores)
        attentions = torch.bmm(self._attention_scores, values)
        
        attention = torch.cat( attentions.chunk(self._heads, dim=0), dim=-1 )
        attention = self._W_o(attention)

        return attention
    
    @property
    def attention_map(self):
        return self._attention_scores

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, d_k: int, d_v: int, heads: int, dropout: float = 0.3):
        super(EncoderLayer, self).__init__()
        self._multihead_self_attention = MultiHeadAttention(d_model, d_k, d_v, heads)
        self._feed_forward = PositionwiseFeedForward(d_model, d_hidden)
        self._add_norm_1 = AddNorm(d_model)
        self._add_norm_2 = AddNorm(d_model)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self._multihead_self_attention(query=x, key=x, value=x)
        x = self._dropout(x)
        x = self._add_norm_1(x, residual)

        residual = x
        x = self._feed_forward(x)
        x = self._dropout(x)
        x = self._add_norm_2(x, residual)

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, d_k: int, d_v: int, heads: int, dropout: float = 0.3):
        super(DecoderLayer, self).__init__()
        self._masked_multihead_self_attention = MultiHeadAttention(d_model, d_k, d_v, heads)
        self._multihead_attention = MultiHeadAttention(d_model, d_k, d_v, heads)
        self._feed_forward = PositionwiseFeedForward(d_model, d_hidden)
        self._add_norm_1 = AddNorm(d_model)
        self._add_norm_2 = AddNorm(d_model)
        self._add_norm_3 = AddNorm(d_model)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        residual = x
        x = self._masked_multihead_self_attention(query=x, key=x, value=x, mask='decoder')
        x = self._dropout(x)
        x = self._add_norm_1(x, residual)

        residual = x
        x = self._multihead_attention(query=x, key=encoder_output, value=encoder_output)
        x = self._dropout(x)
        x = self._add_norm_2(x, residual)

        residual = x
        x = self._feed_forward(x)
        x = self._dropout(x)
        x = self._add_norm_3(x, residual)

        return x

class Transformer(nn.Module):
    def __init__(self, num_layer: int, d_in: int, d_model: int, d_hidden: int, d_out: int, d_k: int, d_v: int, heads: int, dropout: float =0.3):
        super(Transformer, self).__init__()
        self._encoder_embedding_position_encoding = EmbeddingPositionEncoding(d_in, d_model)
        # self._decoder_embedding_position_encoding = EmbeddingPositionEncoding(d_in, d_model)
        self._encoder = nn.ModuleList([EncoderLayer(d_model, d_hidden, d_k, d_v, heads, dropout) for _ in range(num_layer)])
        self._decoder = nn.ModuleList([DecoderLayer(d_model, d_hidden, d_k, d_v, heads, dropout) for _ in range(num_layer)])
        self._output_layer = nn.Linear(d_model, d_out)
        self._activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        encoding = self._encoder_embedding_position_encoding(x)

        for encoder_layer in self._encoder:
            encoding = encoder_layer(encoding)

        decoding = encoding

        for decoder_layer in self._decoder:
            decoding = decoder_layer(decoding, encoding)
        
        output = self._activation(self._output_layer(decoding))
        
        return output

class Generator(nn.Module):
    def __init__(self, num_layer: int, d_in: int, d_model: int, d_hidden: int, d_out: int, d_k: int, d_v: int, heads: int, dropout: float =0.3):
        super(Generator, self).__init__()

        self._transformer = Transformer(num_layer, d_in, d_model, d_hidden, d_out, d_k, d_v, heads, dropout)

    def forward(self, x: torch.Tensor):
        x = self._transformer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, length: int, num_layer: int, d_in: int, d_model: int, d_hidden: int, d_k: int, d_v: int, heads: int, dropout: float = 0.3):
        super(Discriminator, self).__init__()
        
        self._encoder_embedding_position_encoding = EmbeddingPositionEncoding(d_in, d_model)
        self._encoder = nn.ModuleList([EncoderLayer(d_model, d_hidden, d_k, d_v, heads, dropout) for _ in range(num_layer)])
        self._linear_1 = nn.Linear(d_model, 1)
        self._linear_2 = nn.Linear(length, 1)
        self._activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        encoding = self._encoder_embedding_position_encoding(x)

        for encoder_layer in self._encoder:
            encoding = encoder_layer(encoding)
        
        # reduce the dimension of feature
        output = self._activation(self._linear_1(encoding))
        
        # reduce the dimension of sequence
        output = self._linear_2(output.squeeze(-1))

        return output



class BL_dataset(Dataset):
    def __init__(self, path, flag: str = 'train'):
        all_data = pd.read_csv(path)

        group_num = all_data.shape[1] - 1
        test_num = 10
        reshape_data = np.zeros([1,48])
        if flag == 'train':
            for i in range(0, group_num-10):
                temp = all_data.iloc[:, i+1].values.reshape(-1, 48)
                reshape_data = np.concatenate([reshape_data, temp])
        elif flag == 'test':
            for i in range(group_num-10, group_num):
                temp = all_data.iloc[:, i+1].values.reshape(-1, 48)
                reshape_data = np.concatenate([reshape_data, temp])

        reshape_data = np.delete(reshape_data, 0, axis=0)
        self.data = torch.tensor(reshape_data, dtype=torch.float32).unsqueeze(-1)

        print("{} data: {} pieces, [{}, {}], start from {}".format(flag, self.data.shape[0], self.data.shape[1], self.data.shape[2], all_data.iloc[0, 0]))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        return self.data[index]

class FullDataset(Dataset):
    def __init__(self, path, flag, type):
        all_data = pd.read_csv(path)
        if type == 0:
            select_data = all_data.iloc[:, :3]
        elif type == 1:
            select_data = all_data.iloc[:, 3:]
        else:
            select_data = all_data.iloc[:, 1:]
        if flag == 'train':
            self.data = torch.from_numpy(select_data.T.iloc[:, :-122*48].values.reshape(-1, 48)).type(torch.float32).unsqueeze(-1)
        elif flag == 'test':
            self.data = torch.from_numpy(select_data.T.iloc[:, -122*48:].values.reshape(-1, 48)).type(torch.float32).unsqueeze(-1)

        print("{} data: {} pieces, [{}, {}]".format(flag, self.data.shape[0], self.data.shape[1], self.data.shape[2]))

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]



def dr_missing(data, mask):
    flag = np.random.uniform(0, 1)
    if 0 <= flag < 0.2:
        data[10*2+1:12*2+1] = -1.0
        mask[10*2+1:12*2+1] = 0
    elif 0.2 <= flag < 0.4:
        data[14*2+1:17*2+1] = -1.0
        mask[14*2+1:17*2+1] = 0
    elif 0.4 <= flag < 0.6:
        data[14*2+1:17*2+1] = -1.0
        mask[14*2+1:17*2+1] = 0
    elif 0.6 <= flag < 0.7:
        data[10*2+1:12*2+1] = -1.0
        mask[10*2+1:12*2+1] = 0
        data[14*2+1:17*2+1] = -1.0
        mask[14*2+1:17*2+1] = 0
    elif 0.7 <= flag < 0.8:
        data[14*2+1:17*2+1] = -1.0
        mask[14*2+1:17*2+1] = 0
        data[17*2+1:19*2+1] = -1.0
        mask[17*2+1:19*2+1] = 0
    elif 0.8 <= flag < 0.9:
        data[10*2+1:12*2+1] = -1.0
        mask[10*2+1:12*2+1] = 0
        data[17*2+1:19*2+1] = -1.0
        mask[17*2+1:19*2+1] = 0
    elif 0.9 <= flag < 1.0:
        data[10*2+1:12*2+1] = -1.0
        mask[10*2+1:12*2+1] = 0
        data[14*2+1:17*2+1] = -1.0
        mask[14*2+1:17*2+1] = 0
        data[17*2+1:19*2+1] = -1.0
        mask[17*2+1:19*2+1] = 0
    return data, mask

def dr_event(data, dr_rate):
    masks = torch.ones_like(data)
    dr_flag = np.random.uniform(0, 1, data.shape[0]) < dr_rate
    for i, flag in enumerate(dr_flag):
        if flag:
            data[i], masks[i] = dr_missing(data[i], masks[i])
    return data, masks



train_dataset = FullDataset(args.file_path, 'train', 2)
test_dataset = FullDataset(args.file_path, 'test', 2)
# train_dataset = BL_dataset("../data/data_from_2012_10_aggregated_10.csv", 'train')
# test_dataset = BL_dataset("../data/data_from_2012_10_aggregated_10.csv", 'test')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)



model_G = Generator(num_layer=args.num_layer, d_in=args.d_in, d_model=args.d_model, d_hidden=args.d_hidden, d_out=args.d_out, d_k=args.d_key, d_v=args.d_value, heads=args.heads, dropout=args.dropout).to(device)
model_D = Discriminator(length=48, num_layer=args.num_layer, d_in=args.d_in, d_model=args.d_model, d_hidden=args.d_hidden, d_k=args.d_key, d_v=args.d_value, heads=args.heads, dropout=args.dropout).to(device)

optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2))

losses = {
    "model_G": [],
    "model_D": [],
    "gradient_penalty": [],
    "gradient_norm": []
}


epoch_loss = {
    "G": [],
    "D": []
}
for epoch in range(args.num_epoch):
    temp_epoch_model_G = []
    temp_epoch_model_D = []
    temp_epoch_gradient_norm = []
    temp_epoch_gradient_penalty = []

    for _, data in enumerate(train_dataloader):
        data, masks = dr_event(data, 0.3)
        real_data = data.to(device)
        masks = masks.to(device)

        temp_model_D = []
        temp_gradient_penalty = []
        temp_gradient_norm = []
        # discriminator training
        for i in range(args.num_critic):
            fake_data = model_G(real_data).to(device)

            # critic score
            real_score = model_D(real_data)
            fake_score = model_D(fake_data).detach()

            # gradient penalty
            shape = [real_data.shape[i] if i==0 else 1 for i in range(len(real_data.shape))]
            epsilon = torch.rand(shape).to(device)
            interpolated_data = (epsilon * real_data + (1-epsilon) * fake_data).to(device)
            interpolated_score = model_D(interpolated_data)

            # calculate the gradient of interpolated data, not use backward() method
            interpolated_gradient = torch.autograd.grad(outputs=interpolated_score, inputs=interpolated_data, grad_outputs=torch.ones_like(interpolated_score).to(device), retain_graph=True, create_graph=True)[0]

            gradient_norm = interpolated_gradient.reshape(real_data.shape[0], -1).norm(2, dim=-1)
            temp_gradient_norm.append(gradient_norm.mean().item())

            gradient_penalty = args.gradient_penalty_weight * ((gradient_norm-1)**2).mean()
            temp_gradient_penalty.append(gradient_penalty.item())

            # discriminator loss (WGAN-gp)
            loss_D = fake_score.mean() - real_score.mean() + gradient_penalty
            # temp_model_D.append(loss_D.item())
            temp_model_D.append((real_score.mean() - fake_score.mean()).item())

            # update discriminator
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        temp_epoch_model_D.append(temp_model_D)
        temp_epoch_gradient_norm.append(temp_gradient_norm)
        temp_epoch_gradient_penalty.append(temp_gradient_penalty)

        # generator training
        fake_data = model_G(real_data).to(device)

        # critic score
        fake_score = model_D(fake_data)

        # element-wise difference
        element_norm = (real_data*masks - fake_data*masks).norm(2, dim=-1).norm(2, dim=-1).mean()

        # generator loss (data recovery)
        loss_G = args.element_loss_weight * element_norm - fake_score.mean()
        temp_epoch_model_G.append(loss_G.item())

        # update generator
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
    
    losses["model_G"].append(temp_epoch_model_G)
    losses["model_D"].append(temp_epoch_model_D)
    losses["gradient_norm"].append(temp_epoch_gradient_norm)
    losses["gradient_penalty"].append(temp_epoch_gradient_penalty)



    log_model_G = pd.DataFrame(temp_epoch_model_G, columns=['iteration_'+str(i+1) for i in range(1)], index=['batch_'+str(i+1) for i in range(len(train_dataloader))])
    log_model_D = pd.DataFrame(temp_epoch_model_D, columns=['iteration_'+str(i+1) for i in range(args.num_critic)], index=['batch_'+str(i+1) for i in range(len(train_dataloader))])

    log_model_G.to_csv("../log/G/Epoch_"+str(epoch+1)+"_G_loss.csv")
    log_model_D.to_csv("../log/D/Epoch_"+str(epoch+1)+"_D_loss.csv")

    torch.save(model_G.state_dict(), "../log/model/G.pt")
    torch.save(model_D.state_dict(), "../log/model/D.pt")
    
    print("Epoch {}/{}, Generator Loss: {}, Discriminator Loss: {}".format(epoch+1, args.num_epoch, np.array(losses["model_G"][-1], dtype=float).mean(), np.array(losses["model_D"][-1], dtype=float).mean()))

    epoch_loss["G"].append(np.array(losses["model_G"][-1], dtype=float).mean())
    epoch_loss["D"].append(np.array(losses["model_D"][-1], dtype=float).mean())

training_loss = pd.DataFrame(epoch_loss)
training_loss.to_csv("../result/training_loss.csv", index=None)



model_G.eval()
real_data = []
meter_data = []
generated_data = []

for _, data in enumerate(test_dataloader):
    real_data.append(data.flatten().tolist())
    
    data, masks = dr_event(data, 0.3)
    data = data.to(device)
    masks = masks.to(device)
    meter_data.append(data.flatten().tolist())

    data = data.to(device)
    fake_data = model_G(data).to(device)
    generated_data.append(fake_data.cpu().detach().flatten().tolist())

np.save("../result/real_data.npy", np.array(real_data))
np.save("../result/meter_data.npy", np.array(meter_data))
np.save("../result/generated_data.npy", np.array(generated_data))