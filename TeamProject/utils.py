import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def display_one_data(df, node_num = 200, day = 30, save=False, title=None):
    temp = list(df.groupby(['start_node_name', 'end_node_name']))[node_num][1]
    plt.plot(temp['target'][24 * day: 24 * (day+1)].tolist())
    # plt.title(f'Target values of a node within 24 hours')
    plt.xticks(np.arange(0,24))
    if save:
        plt.savefig(f'{title}.png')

def make_time_series_data(df, T = 24, t = 24):
    
    # Make time series data, i.e., time s data has information of time s-1, s-2, ..., s-T.
    # And, the model predicts s, s+1, ..., s+t.

    lst = []
    for x in df.groupby(['start_node_name', 'end_node_name']):
        temp = x[1]
        lst_target = temp['target'].tolist()
        temp_ = temp[T:-t]
        lst_pre = []
        lst_fut = []
        for i in range(len(temp_)):
            lst_pre.append(lst_target[i:i+T])
            lst_fut.append(lst_target[i+T:i+T+t])
        temp_['Past_time_series'] = lst_pre
        temp_['Target_time_series'] = lst_fut
        lst.append(temp_)
    df_T = pd.concat(lst).sort_values(['base_date', 'base_hour', 'start_node_name', 'end_node_name'])

    return df_T

def make_edge_relationship(df):
    # make edge relationship
    N = df.groupby(['start_node_name', 'end_node_name']).__len__()
    test = df.copy().sort_values(['base_date', 'base_hour', 'start_node_name', 'end_node_name']).reset_index()[:N]

    lst_edges = []
    for i in range(N):
        s = test.iloc[i]['start_node_name']
        e = test.iloc[i]['end_node_name']
        targets = sorted(set(test[test['end_node_name'] == s].index.tolist() + test[test['start_node_name'] == e].index.tolist()))
        for j in targets:
            lst_edges.append([i, j])
    
    return lst_edges

def del_feature_(x, d, del_feature):
    dict_ = {'day_of_week' : torch.arange(0, 7), 
             'lane_count' : torch.arange(7, 10), 
             'road_rating': torch.arange(10, 13), 
             'road_type' : torch.arange(13, 15), 
             'end_turn_restricted' : torch.arange(15, 17), 
             'start_latitude' : torch.arange(17, 19), 
            #  'start_longitude' : torch.arange(18, 19), 
             'end_latitude': torch.arange(19, 21), 
            #  'end_longitude': torch.arange(20, 21),
             'start+end' : torch.arange(17, 21),
             }
    if del_feature not in dict_.keys():
        d = 1
        return x, d
    else:
        d = 2
        if del_feature in ['day_of_week', 'lane_count', 'road_rating', 'road_type', 'end_turn_restricted']:
            x[:, dict_[del_feature]] = 0
        if del_feature in ['start_latitude', 'end_latitude']:
            x[:, dict_[del_feature]] = torch.rand(x.size(0), 2)
        if del_feature in ['start+end']:
            x[:, dict_[del_feature]] = torch.rand(x.size(0), 4)
        return x, d

def make_dataloader(df, edge_index, device, T = 24, del_feature = None):
    dataloader = []
    d = 0
    for tup, data in df.groupby(['base_date', 'base_hour']):
        
        day_of_week = torch.zeros((564,7))
        day_of_week[:, data.iloc[0]['day_of_week']] = 1
            
        cat_data = torch.concat([torch.nn.functional.one_hot(torch.tensor(data[cat].to_list()).to(int)) for cat in ['lane_count', 'road_rating', 'road_type', 'end_turn_restricted']], dim=1)
        num_data = torch.Tensor(data[['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']].to_numpy())

        x = torch.concat([day_of_week, cat_data, num_data], dim=1)
        if del_feature is not None:
            x, d = del_feature_(x, d, del_feature)

        x = torch.concat((
                x.unsqueeze(-1).repeat(1, 1, T), 
                torch.Tensor(np.vstack(data['Past_time_series'].to_numpy())).unsqueeze(1)
                ), dim=1).to(torch.float)
        
        y = torch.Tensor(np.vstack(data['Target_time_series'].to_numpy()))

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
        data = Data(x=x.to(device), edge_index=edge_index.t().contiguous(), y=y.to(device))
        dataloader.append(data)
    
    if d == 1:
        # print(f'Feature deletion did not work on dataloader_1')
    if d == 2:
        # print(f'Feature deletion did work on {del_feature}')
    return dataloader
