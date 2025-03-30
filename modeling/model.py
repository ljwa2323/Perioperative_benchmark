import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()
        self.init_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        output = self.layer2(x)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # 输出层
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        # LSTM 输出
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 输出层
        output = self.fc_out(last_output)
        
        return output

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 双向 LSTM 参数
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # 输出层
        self.fc_out = nn.Linear(2 * hidden_size, output_size)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM 输出
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, 2 * hidden_size]
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 输出层
        output = self.fc_out(last_output)
        
        return output


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMWithAttention, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 双向 LSTM 参数
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Attention 层
        self.attention = nn.Linear(2 * hidden_size, 1)
        
        # 输出层
        self.fc_out = nn.Linear(2 * hidden_size, output_size)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM 输出
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, 2 * hidden_size]
        
        # Attention 权重
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # 加权求和得到上下文向量
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 输出层
        output = self.fc_out(context_vector)
        
        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 使用整个 GRU 层而不是 GRUCell
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        # 输出层
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        # GRU 输出
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_len, hidden_size]
        
        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]
        
        # 输出层
        output = self.fc_out(last_output)
        
        return output
    
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 双向 GRU 参数
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # 输出层
        self.fc_out = nn.Linear(2 * hidden_size, output_size)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # GRU 输出
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_len, 2 * hidden_size]
        
        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]
        
        # 输出层
        output = self.fc_out(last_output)
        
        return output

class BiGRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRUWithAttention, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 双向 GRU 参数
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # 注意力层
        self.attention = nn.Linear(2 * hidden_size, 1)
        
        # 输出层
        self.fc_out = nn.Linear(2 * hidden_size, output_size)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # GRU 输出
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_len, 2 * hidden_size]
        
        # 注意力权重
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)
        
        # 加权求和得到上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        
        # 输出层
        output = self.fc_out(context_vector)
        
        return output

class PredModel(nn.Module):
    def __init__(self, input_size_list, hidden_size_list, model_list, \
                 output_size_list, type_list):
        super().__init__()
        self.models = nn.ModuleList()
        for i, model_name in enumerate(model_list):
            model_class = MODEL_LIST[model_name]
            self.models.append(model_class(input_size_list[i], hidden_size_list[i], input_size_list[i]))

        self.input_size_list = input_size_list
        self.hidden_size_list = hidden_size_list
        self.model_list = model_list
        self.output_size_list = output_size_list
        self.type_list = type_list
        
        self.pred_module_list = nn.ModuleList()
        for i, output_size in enumerate(output_size_list):
            if type_list[i] == "cat":
                if output_size == 1:
                    self.pred_module_list.append(nn.Sequential(MLP(sum(self.input_size_list), 200, output_size), nn.Sigmoid()))
                elif output_size >= 2:
                    self.pred_module_list.append(nn.Sequential(MLP(sum(self.input_size_list), 200, output_size), nn.Softmax(dim=-1)))
            elif type_list[i] == "num":
                self.pred_module_list.append(MLP(sum(self.input_size_list), 200, output_size))

    def forward(self, x_list):

        hidden_list = []
        for i, model_name in enumerate(self.model_list):
            if model_name == 'mlp':
                output = self.models[i](x_list[i]).squeeze(1)
            elif model_name in ['lstm',"gru","bilstm",'bigru','biattlstm','biattgru'] :
                output = self.models[i](x_list[i])
            hidden_list.append(output)

        hiddens = torch.cat(hidden_list, dim=-1)

        output_list = []
        for pred_module in self.pred_module_list:
            output = pred_module(hiddens)
            output_list.append(output)
        return output_list

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

MODEL_LIST = {"mlp":MLP, "gru": GRU,
                "bigru":  BiGRU,
                "biattgru" : BiGRUWithAttention,
                "lstm": LSTM,
                "bilstm" : BiLSTM, 
                "biattlstm" : BiLSTMWithAttention}

if __name__ == "__main__":

    from model import *
    from tqdm import tqdm
    import numpy as np

    x_list = [torch.randn(1, 4, 20)]
    m_list = [torch.randint(0, 2, (1, 4, 20))]
    t_list = [torch.arange(0, 4).reshape(1, -1, 1)]

    input_size_list = [20]
    hidden_size_list = [128]
    output_size_list = [1, 1]
    type_list = ["cat","num"]
    model_list = ["biattlstm"]

    model = PredModel(input_size_list, hidden_size_list, model_list, output_size_list, type_list)

    yhat = model(x_list)

    # print("--")