D-VAE -- A Variational Autoencoder for Directed Acyclic Graphs
===============================================================================

About
-----

Directed acyclic graphs (DAGs) are of particular interest to machine learning researchers, as many machine learning models are realized as computations on DAGs, including neural networks and Bayesian networks. Two important problems, neural architecture search (NAS) and Bayesian network structure learning (BNSL), are essentially DAG optimization problems, where an optimal DAG structure is to be found to best fit the given dataset.

D-VAE is a variational autoencoder for DAGs. It encodes/decodes DAGs using an asynchronous message passing scheme, and is able to injectively encode computations on DAGs. D-VAE provides a new direction for DAG optimization. By embedding DAGs into a continuous latent space, D-VAE transforms the difficult discrete optimization problem into an easier continuous space optimization problem, where principled Bayesian optimization can be performed in this latent space to optimize DAG structures.

For more information, please check our paper:
> M. Zhang, S. Jiang, Z. Cui, R. Garnett, Y. Chen, D-VAE: A Variational Autoencoder for Directed Acyclic Graphs. [\[Preprint\]](https://arxiv.org/pdf/1904.11088.pdf)

Installation
------------

Tested with Python 3.6, PyTorch 0.4.1.

Install [PyTorch](https://pytorch.org/) >= 0.4.1

Install python-igraph by:

    pip install python-igraph

Install pygraphviz by:

    conda install graphviz
    conda install pygraphviz

Other required python libraries: tqdm, six, scipy, numpy, matplotlib

Training
--------

### Neural Architectures

    python train.py --data-name final_structures6 --save-interval 100 --save-appendix _DVAE --epochs 300 --lr 1e-4 --model DVAE --nz 56 --batch-size 32

### Bayesian Networks

    python train.py --data-name asia_200k --data-type BN --nvt 8 --save-interval 50 --save-appendix _DVAE --epochs 100 --lr 1e-4 --model DVAE_BN --nz 56 --batch-size 128

Bayesian Optimization
---------------------

To perform Bayesian optimization experiments after training D-VAE, the following additional steps are needed.

Install sparse Gaussian Process (SGP) based on Theano:

    cd bayesian_optimization/Theano-master/
    python setup.py install
    cd ../..

Download the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by: 

    cd software/enas
    mkdir data
    cd data
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    mv cifar-10-batches-py/ cifar10/

Install [TensorFlow](https://www.tensorflow.org/install/gpu) >= 1.12.0

Install R package _bnlearn_:

    R
    install.packages('bnlearn', lib='/R/library', repos='http://cran.us.r-project.org')

Then, in "bayesian_optimization/", type:

    ./run_bo_NN.sh

and 

    ./run_bo_BN.sh

to run Bayesian optimization for neural architecturs and Bayesian networks, respectively.

Finally, to summarize the BO results, type:

    python summarize.py

The results will be saved in "bayesian_optimization/**_aggregate_results/". The settings can be changed within "summarize.py".

Reference
---------

If you find the code useful, please cite our paper.

    @article{zhang2019d,
      title={D-VAE: A Variational Autoencoder for Directed Acyclic Graphs},
      author={Zhang, Muhan and Jiang, Shali and Cui, Zhicheng and Garnett, Roman and Chen, Yixin},
      journal={arXiv preprint arXiv:1904.11088},
      year={2019}
    } 

Muhan Zhang, Washington University in St. Louis
muhan@wustl.edu
5/13/2019
