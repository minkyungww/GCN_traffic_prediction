<img src="/imgs/graphical_model.png" width="100%" height="100%">

# Forecasting Real-time Traffic Information Based on Historical Traffic Data
This repository is dedicated to the results of the DS535 class in the fall of 2023. We utilized GNN for traffic forecasting while simultaneously incorporating hierarchical information of administrative areas to propose a novel model. A summarized meterial of the project can be found in the [G4_DS535_final_presentation](https://github.com/GwangWooKim/RecSys-GNN/blob/main/G4_DS535_final_presentation.pdf). The final report is available [here](https://www.overleaf.com/read/jzbyzqgfywbb#cdb988).

## Setup for reproducing our results
* python == 3.10.12
* pytorch == 2.1.1
* torch_geometric == 2.4.0
* torch-geometric-temporal == 0.54.0
* matplotlib
* pyarrow

In order to reproduce our results, create a new environment with `python == 3.10.12` and run `setup.sh`. For example, 
```
$ sh setup_cuda121.sh
```
:fire:**Warning: The shall script is running on cuda 12.1, so modify `setup_cuda121.sh` file up to your local machine.**:fire:

## How to run
All arguments of the implementation are set to reproduce the results of the presentation. We highly recommend running with GPU and using the suggested values that were optimized by our study.

### Example
    $ python main.py
* `-s` : Seed. `Default = 42`.
* `-t` : The length of the prediction interval. `Default = 1`.
* `-T` : The length of the input timesteps. `Default = 24`.
* `-e` : Epochs. `Default = 100`.
* `--lr` : Learning rate. `Default = 0.001`.
* `--del_feature` (or `-d`): The specified feature will not be used to train the model. `Default = None`.

### Description of the outputs
Once trained the model on the given dataset, you will obtain the following files (e.g., in dir: ./output/model_del_feature=None) and can check them via `torch.load`.
* `model_best.pt` : the parameters of the trained model.
* `model_step_size.pt` : the log of learning rate during training.
* `model_train_loss.pt` : the log of training loss.
* `model_val_loss.pt` : the log of validation loss. 

## Evaluation

One example to visualize the training log is as follows:

```python
import torch
import matplotlib.pyplot as plt

path = './output/model_del_feature=None' # modify the path if necessary
lst_train_loss = torch.load(path + '/model_train_loss.pt')
lst_val_loss = torch.load(path + '/model_val_loss.pt')
lst_step_size = torch.load(path + '/model_step_size.pt')

plt.figure()
plt.plot(lst_train_loss, label = 'Train')
plt.plot(lst_val_loss, label = 'Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(lst_step_size)
plt.xlabel('Epoch')
plt.ylabel('lr')
plt.show()
```

<img src="/imgs/training_log.png" width="65%" height="65%">
<img src="/imgs/step_size.png" width="65%" height="65%">

If you want to visualize quantitative comparisons or prediction results, please refer to the [inference.ipynb](https://github.com/GwangWooKim/RecSys-GNN/blob/main/TeamProject/inference.ipynb). Quantitative comparisons involve comparing MAE, MSE, and R2 score on the test_dataset with the baseline (LGBM and LGBM+). The visualization includes plotting predictions for each node (or intersection) over a period of 7 days. For example:

<img src="/imgs/one_node.png" width="65%" height="65%">

## Miscellanea

We conducted not only the mentioned studies but also ablation study and feature importance study. Since these are very tedious processes, they are not implemented in this repository. In fact, you can conduct the ablation study by changing various models in `models.py`. The feature importance study can be done by inputting features one by one to `del_feature`. Please refer to [G4_DS535_final_presentation](https://github.com/GwangWooKim/RecSys-GNN/blob/main/G4_DS535_final_presentation.pdf) for detailed results.
