import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Controller(nn.Module):
    def __init__(self, embedding_size, hidden_size=100, device="cpu", seed=2):
        super(Controller, self).__init__()
        self.embedding_size = embedding_size
        self.num_exps = 3
        self.num_ker = 2
        self.num_l = 3
        self.num_se = 2
        self.seed = seed
        self.num_actions = self.num_exps + self.num_ker + self.num_l + 1 + self.num_se

        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(self.num_actions, self.embedding_size)

        self.e_decoder = nn.Linear(hidden_size, self.num_exps)
        self.k_decoder = nn.Linear(hidden_size, self.num_ker)
        self.l_decoder_1 = nn.Linear(hidden_size, self.num_l)
        self.l_decoder_2 = nn.Linear(hidden_size, self.num_l + 1)
        self.se_decoder = nn.Linear(hidden_size, self.num_se)

        self.rnn = nn.LSTMCell(self.embedding_size, hidden_size)

        self.init_parameters()

    def forward(self, input, h_t, c_t, decoder):
        input = self.embedding(input)
        h_t, c_t = self.rnn(input, (h_t, c_t))
        logits = decoder(h_t)
        return h_t, c_t, logits

    def sample(self):
        input = torch.LongTensor([self.seed]).to(self.device)
        h_t, c_t = self.init_hidden()
        actions_p = []
        actions_log_p = []
        actions_index = []

        for lay in range(7):
            # Exp Factor
            h_t, c_t, logits = self.forward(input, h_t, c_t, self.e_decoder)
            action_index = Categorical(logits=logits).sample()
            # if lay == 0:
            #    action_index = torch.tensor([2]).to('cuda:0')
            p = F.softmax(logits, dim=-1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0, action_index]
            actions_p.append(p.detach())
            actions_log_p.append(log_p.detach())
            actions_index.append(action_index)

            # Kernel Size
            input = action_index
            h_t, c_t, logits = self.forward(input, h_t, c_t, self.k_decoder)
            action_index = Categorical(logits=logits).sample()
            p = F.softmax(logits, dim=-1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0, action_index]
            actions_p.append(p.detach())
            actions_log_p.append(log_p.detach())
            actions_index.append(action_index)

            # Number of Layers
            input = action_index + self.num_exps
            h_t, c_t, logits = self.forward(
                input, h_t, c_t, self.l_decoder_1 if lay != 5 else self.l_decoder_2
            )
            action_index = Categorical(logits=logits).sample()
            # if lay == 0:
            #    action_index = torch.tensor([2]).to('cuda:0')
            p = F.softmax(logits, dim=-1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0, action_index]
            actions_p.append(p.detach())
            actions_log_p.append(log_p.detach())
            actions_index.append(action_index)

            # SE State
            input = action_index + self.num_exps + self.num_ker
            h_t, c_t, logits = self.forward(input, h_t, c_t, self.se_decoder)
            action_index = Categorical(logits=logits).sample()
            # if lay == 0:
            #    action_index = torch.tensor([1]).to('cuda:0')
            p = F.softmax(logits, dim=-1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0, action_index]
            actions_p.append(p.detach())
            actions_log_p.append(log_p.detach())
            actions_index.append(action_index)

            input = action_index + self.num_exps + self.num_ker + self.num_l + 1

        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)
        actions_index = torch.cat(actions_index)
        return actions_p, actions_log_p, actions_index

    def get_p(self, actions_index):
        input = torch.LongTensor([self.seed]).to(self.device)
        h_t, c_t = self.init_hidden()
        t = 0
        actions_p = []
        actions_log_p = []

        for lay in range(7):
            h_t, c_t, logits = self.forward(input, h_t, c_t, self.e_decoder)
            action_index = actions_index[t].unsqueeze(0)
            t += 1
            p = F.softmax(logits, dim=-1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0, action_index]
            actions_p.append(p)
            actions_log_p.append(log_p)

            input = action_index
            h_t, c_t, logits = self.forward(input, h_t, c_t, self.k_decoder)
            action_index = actions_index[t].unsqueeze(0)
            t += 1
            p = F.softmax(logits, dim=-1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0, action_index]
            actions_p.append(p)
            actions_log_p.append(log_p)

            # Number of Layers
            input = action_index + self.num_exps
            h_t, c_t, logits = self.forward(
                input, h_t, c_t, self.l_decoder_1 if lay != 5 else self.l_decoder_2
            )
            action_index = actions_index[t].unsqueeze(0)
            t += 1
            p = F.softmax(logits, dim=-1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0, action_index]
            actions_p.append(p)
            actions_log_p.append(log_p)

            # SE State
            input = action_index + self.num_exps + self.num_ker
            h_t, c_t, logits = self.forward(input, h_t, c_t, self.se_decoder)
            action_index = actions_index[t].unsqueeze(0)
            t += 1
            p = F.softmax(logits, dim=-1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0, action_index]
            actions_p.append(p)
            actions_log_p.append(log_p)

            input = action_index + self.num_exps + self.num_ker + self.num_l + 1

        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)

        return actions_p, actions_log_p

    def init_hidden(self):
        h_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)
        c_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)

        return (h_t, c_t)

    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.e_decoder.bias.data.fill_(0)
        self.k_decoder.bias.data.fill_(0)
        self.l_decoder_1.bias.data.fill_(0)
        self.l_decoder_2.bias.data.fill_(0)
        self.se_decoder.bias.data.fill_(0)
