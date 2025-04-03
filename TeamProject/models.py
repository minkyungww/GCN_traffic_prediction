import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

class TemporalGNN_vanilla(nn.Module):
    def __init__(self, **kwargs):
        super(TemporalGNN_vanilla, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        # add_layers = []
        # for i in range(len(self.lst_channels)-1):
        #     add_layers.append(A3TGCN(in_channels=self.lst_channels[i], out_channels=self.lst_channels[i+1], periods=self.t))
        #     add_layers.append(nn.ReLU())
        
        # For low-level forward process
        self.main_1 = A3TGCN(in_channels=self.lst_channels[0], out_channels=self.lst_channels[1], periods=self.t)
        # self.main_2 = A3TGCN(in_channels=self.lst_channels[1]+1, out_channels=self.lst_channels[2], periods=self.t)

        # For high-level forward process
        # self.main_3 = A3TGCN(in_channels=self.second_dim + self.lst_channels[1] + 1, out_channels=self.lst_channels[2], periods=self.t)

        # Common activation
        self.act = nn.ReLU()

        # final fully-connected layers
        self.linear_1 = nn.Linear(self.lst_channels[1], self.t)
        # self.linear_2 = nn.Linear(self.lst_channels[2], self.t)

    def groupby(self, tensor, labels):
        # It works like pandas.groupby
        # Example: https://stackoverflow.com/questions/56154604/groupby-aggregate-mean-in-pytorch
        M = torch.zeros(labels.max()+1, len(tensor)).to(self.device)
        M[labels, torch.arange(len(tensor))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        return torch.mm(M, tensor)

    def forward(self, x_1, edge_index_1, x_2, edge_index_2):                            
        h = self.main_1(x_1, edge_index_1)                                     
        h = self.act(h)     
        pred = self.linear_1(h)    
        return pred
    
class TemporalGNN_vanilla_two_layer(nn.Module):
    def __init__(self, **kwargs):
        super(TemporalGNN_vanilla_two_layer, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        # add_layers = []
        # for i in range(len(self.lst_channels)-1):
        #     add_layers.append(A3TGCN(in_channels=self.lst_channels[i], out_channels=self.lst_channels[i+1], periods=self.t))
        #     add_layers.append(nn.ReLU())
        
        # For low-level forward process
        self.main_1 = A3TGCN(in_channels=self.lst_channels[0], out_channels=self.lst_channels[1], periods=self.t)
        self.main_2 = A3TGCN(in_channels=self.lst_channels[1]+1, out_channels=self.lst_channels[2], periods=self.t)

        # For high-level forward process
        # self.main_3 = A3TGCN(in_channels=self.second_dim + self.lst_channels[1] + 1, out_channels=self.lst_channels[2], periods=self.t)

        # Common activation
        self.act = nn.ReLU()

        # final fully-connected layers
        self.linear_1 = nn.Linear(self.lst_channels[2], self.lst_channels[1])
        self.linear_2 = nn.Linear(self.lst_channels[1], self.t)

    def groupby(self, tensor, labels):
        # It works like pandas.groupby
        # Example: https://stackoverflow.com/questions/56154604/groupby-aggregate-mean-in-pytorch
        M = torch.zeros(labels.max()+1, len(tensor)).to(self.device)
        M[labels, torch.arange(len(tensor))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        return torch.mm(M, tensor)

    def forward(self, x_1, edge_index_1, x_2, edge_index_2):
        time = x_1[:, -1, :]

        h = self.main_1(x_1, edge_index_1)                                     
        h = self.act(h)   

        h = torch.concat((h.unsqueeze(-1).repeat(1, 1, self.T), time.unsqueeze(1)), dim=1)  

        h = self.main_2(h, edge_index_1)                                    
        h = self.act(h)     
   
        h = self.linear_1(h)
        h = self.act(h)

        pred = self.linear_2(h)    

        return pred

class TemporalGNN(nn.Module):
    def __init__(self, **kwargs):
        super(TemporalGNN, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        # add_layers = []
        # for i in range(len(self.lst_channels)-1):
        #     add_layers.append(A3TGCN(in_channels=self.lst_channels[i], out_channels=self.lst_channels[i+1], periods=self.t))
        #     add_layers.append(nn.ReLU())
        
        # For low-level forward process
        self.main_1 = A3TGCN(in_channels=self.lst_channels[0], out_channels=self.lst_channels[1], periods=self.t)
        self.main_2 = A3TGCN(in_channels=self.lst_channels[1]+1, out_channels=self.lst_channels[2], periods=self.t)

        # For high-level forward process
        self.main_3 = A3TGCN(in_channels=self.second_dim + self.lst_channels[1] + 1, out_channels=self.lst_channels[2], periods=self.t)

        # Common activation
        self.act = nn.ReLU()

        # final fully-connected layers
        self.linear_1 = nn.Linear(2 * self.lst_channels[2], self.lst_channels[2])
        self.linear_2 = nn.Linear(self.lst_channels[2], self.t)

    def groupby(self, tensor, labels):
        # It works like pandas.groupby
        # Example: https://stackoverflow.com/questions/56154604/groupby-aggregate-mean-in-pytorch
        M = torch.zeros(labels.max()+1, len(tensor)).to(self.device)
        M[labels, torch.arange(len(tensor))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        return torch.mm(M, tensor)

    def forward(self, x_1, edge_index_1, x_2, edge_index_2):
        # x_1 : (N, P, T)
        # x_2 : (M, Q)

        ## Memorize time information
        #
        # (N, T)
        time = x_1[:, -1, :]

        ## Common forward process   
        #  
        # (N, P, T) -> (N, lst_channels[1])                              
        h = self.main_1(x_1, edge_index_1)                                     
        h = self.act(h)   

        ## high-level forward process
        #
        # (N, lst_channels[1]) -> (M, lst_channels[1])
        h_ = h.clone() 
        h_ = (self.groupby(h_, labels = self.address_start) + self.groupby(h_, labels = self.address_end)) / 2

        # (N, T) -> (M, T)
        h_time = (self.groupby(time, labels = self.address_start) + self.groupby(time, labels = self.address_end)) / 2

        # (M, Q) + (M, lst_channels[1]) -> (M, Q + lst_channels[1])
        x_2 = torch.concat((x_2, h_), dim=1)

        # x_2 : (M, Q + lst_channels[1]) -> (M, Q + lst_channels[1], 1) -> (M, Q + lst_channels[1], T)
        # h_time: (M, T) -> (M, 1, T)
        # x_2 = x_2 + h_time, that is, (M, Q + lst_channels[1], T) + (M, 1, T) -> (M, Q + lst_channels[1] + 1, T)
        x_2 = torch.concat((x_2.unsqueeze(-1).repeat(1, 1, self.T), h_time.unsqueeze(1)), dim=1)

        # (M, Q + lst_channels[1] + 1, T) -> (M, lst_channels[2])
        h_ = self.main_3(x_2, edge_index_2)
        h_ = self.act(h_)   

        ## low-level forward process
        #
        # h : (N, lst_channels[1]) -> (N, lst_channels[1], 1) -> (N, lst_channels[1], T)  
        # time : (N, T) -> (N, 1, T)
        # h = h + time, that is, (N, lst_channels[1], T) + (N, 1, T) -> (N, lst_channels[1] + 1, T)
        h = torch.concat((h.unsqueeze(-1).repeat(1, 1, self.T), time.unsqueeze(1)), dim=1)  

        # (N, lst_channels[1] + 1, T) -> (N, lst_channels[2])
        h = self.main_2(h, edge_index_1)                                    
        h = self.act(h)     

        ## Combine two hidden variables and do inference.    
        ## Note that h_.shape[0] = M, not N.
        ## In otder to combine them, we have to make h_ to be h_.shape[0] = N.
        ## One way to do is to duplicate the tensors of h_ and rearrange them.  
        #
        # (M, lst_channels[2]) -> (N, lst_channels[2])
        h_ = (torch.concat([h_[index].unsqueeze(0) for index in self.address_start]) + torch.concat([h_[index].unsqueeze(0) for index in self.address_end])) / 2

        # (N, lst_channels[2]) + (N, lst_channels[2]) -> (N, 2 * lst_channels[2]) -> (N, lst_channels[2])              
        h = self.linear_1(torch.concat((h, h_), dim=1))
        h = self.act(h)

        # (N, lst_channels[2]) -> (N, t)   
        pred = self.linear_2(h)    

        return pred
    