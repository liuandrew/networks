This is the ReadMe for the models and functions associated with the paper:

Andrew Liu and Mason A. Porter, 2020,
"Spatial strength centrality and the effect of 
spatial embeddings on network architecture"
Physical Review E.


Files:
plot_examples.py
    Running this file will generate networks to draw many of the plots found
    in the paper
plot_examples.ipynb
    This does the same as the Python file, but is in a Jupyter Notebook format
    which may make viewing the code and interacting with it easier


base_network_extensions.py
    This file contains some base functions used to generate networks
geographical_spatial_strength_extensions.py
    This file contains function to generate each of the 3 models in the paper
    along with functions to compute spatial strength centrality and plot
    the networks
example_spatial_networks.py
    This file contains functions to generate the networks found in Section V C
    and Section V D

target_threshold.pickle
    This file contains the values for theta (threshold parameter) used in the
    GF networks for different values of beta

datasets (folder)
    This folder contains a few of the data files used in Section V D to generate
    empirical data networks. For complete datasets, please refer to
    Fungal Networks: S.H.Lee, M.D.Fricker, M.A.Porter 2017
    Street Networks: S.H.Lee, P. Holme 2012


Python running notes:
These Python files use two potentially nonstandard packages, NetworkX and Seaborn
To ensure that you have the necessary packages to run this code, please use pip
to install the relevant packages to your computer. You can run the following line
in your terminal

$ pip install numpy scipy matplotlib networkx seaborn


We hope you find this code helpful!

--Andrew Liu and Mason A. Porter
