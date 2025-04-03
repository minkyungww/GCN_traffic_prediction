# install pytorch==2.1.1
# take care about cuda version
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# install torch_geometric==2.4.0
conda install pyg -c pyg

# install additional packages of torch_geometric 
# This depends on the pytorch version and cuda version
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# install torch-geometric-temporal==0.54.0
pip install torch-geometric-temporal

# install matplotlib
conda install matplotlib

# install pyarrow
conda install -c conda-forge pyarrow
