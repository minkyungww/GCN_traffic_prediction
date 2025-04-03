from tqdm import tqdm
import time

import torch

def train_GNN(model, train_dataset, val_dataset, optimizer, criterion, scheduler, save_path, args):

    lst_train_loss = []
    lst_val_loss = []
    lst_step_size = []
    val_loss = 100
    threshold = 15

    for epoch in tqdm(range(args.epoch), desc='epoch', position=0):
        model.train()
        lst_train_step_loss = []

        for step, (snapshot_1, snapshot_2) in tqdm(enumerate(train_dataset), desc='train_step', position=1, leave=False):
            optimizer.zero_grad()

            y_hat = model(snapshot_1.x, snapshot_1.edge_index, snapshot_2.x, snapshot_2.edge_index) # Get model predictions
            loss = criterion(y_hat, snapshot_1.y) # Mean squared error
            # loss, _, _ = dilate_loss(y_hat.unsqueeze(-1), snapshot_1.y.unsqueeze(-1), 0.5, 2, device)

            loss.backward()
            optimizer.step()

            lst_train_step_loss.append(loss.item())

        lst_step_size.append(scheduler.get_last_lr())
        scheduler.step()
        lst_train_loss.append(sum(lst_train_step_loss) / step)

        with torch.inference_mode():

            lst_val_step_loss = []
        
            for step, (snapshot_1, snapshot_2) in tqdm(enumerate(val_dataset), desc='val_step', position=2, leave=False):
                optimizer.zero_grad()
                
                y_hat = model(snapshot_1.x, snapshot_1.edge_index, snapshot_2.x, snapshot_2.edge_index)
                loss = criterion(y_hat, snapshot_1.y)

                lst_val_step_loss.append(loss.item())

            lst_val_loss.append(sum(lst_val_step_loss) / step)

            ## Early Stopping
            if val_loss > lst_val_loss[-1]:
                val_loss = lst_val_loss[-1]
                tor = 0
                torch.save(model.state_dict(), save_path + f'/model_epoch={epoch}.pt')
                torch.save(model.state_dict(), save_path + f'/model_best.pt')
            else:
                tor += 1
                if tor >  threshold:
                    print(f'Early Stopping: There was no improvement during last {threshold} epochs')
                    break 

    torch.save(model.state_dict(), save_path + f'/model_epoch={epoch}_last.pt')
    torch.save(lst_train_loss, save_path + '/model_train_loss.pt')
    torch.save(lst_val_loss, save_path + '/model_val_loss.pt')
    torch.save(lst_step_size, save_path + '/model_step_size.pt')