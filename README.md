D-VAE -- A Variational Autoencoder for Directed Acyclic Graphs
===============================================================================

About
-----

Directed acyclic graphs (DAGs) are of particular interest to machine learning researchers, as many machine learning models are realized as computations on DAGs, including neural networks and Bayesian networks. Two important problems, neural architecture search (NAS) and Bayesian network structure learning (BNSL), are essentially DAG optimization problems, where an optimal DAG structure is to be found to best fit a given dataset.

D-VAE is a variational autoencoder for DAGs. It encodes/decodes DAGs using an asynchronous message passing scheme where a node updates its state only after all its predecessors' have been updated. The final node's state can injectively encode the computation on a DAG, rather than only encoding local structures as in standard simultaneous message passing. After training on some DAG distribution, D-VAE can not only generate novel and valid DAGs, but also be used to optimize DAG structures in its latent space. By embedding DAGs into a continuous latent space, D-VAE transforms the difficult discrete optimization problem into an easier continuous space optimization problem, where principled Bayesian optimization can be performed in this latent space to optimize DAG structures. Thanks to the computation-encoding property, D-VAE also empirically embeds DAGs with similar computation purposes (and performances) into the same region, which greatly facilitates the Bayesian optimization.

For more information, please check our paper:
> M. Zhang, S. Jiang, Z. Cui, R. Garnett, Y. Chen, D-VAE: A Variational Autoencoder for Directed Acyclic Graphs, Advances in Neural Information Processing Systems (NeurIPS-19). [\[PDF\]](https://arxiv.org/pdf/1904.11088.pdf)

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

    python train.py --data-name final_structures6 --save-interval 100 --save-appendix _DVAE --epochs 300 --lr 1e-4 --model DVAE --bidirectional --nz 56 --batch-size 32

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
    cd ../..

Download the 6-layer [pretrained ENAS model](https://drive.google.com/drive/folders/1e-mYRZS_10Aegj8Sczcb948RbHyiju1S?usp=sharing) to "software/enas/" (for evaluating a neural architecture's weight-sharing accuracy). There should be a folder named "software/enas/outputs_6/", which contains four model files. The 12-layer pretrained ENAS model is available [here](https://drive.google.com/drive/folders/18GU9g5DNiHn2MOVKOiF1fCwNQMTA-mnH?usp=sharing) too.

Install [TensorFlow](https://www.tensorflow.org/install/gpu) >= 1.12.0

Install R package _bnlearn_:

    R
    install.packages('bnlearn', lib='/R/library', repos='http://cran.us.r-project.org')

Then, in "bayesian_optimization/", type:

    ./run_bo_ENAS.sh

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
      booktitle={Advances in Neural Information Processing Systems},
      pages={1586--1598},
      year={2019}
    } 

Muhan Zhang, Washington University in St. Louis
muhan@wustl.edu
5/13/2019
