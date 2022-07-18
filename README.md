# Code accompanying the arXiv preprint [NuSPAN: A Proximal Average Network for Nonuniform Sparse Model -- Application to Seismic Reflectivity Inversion](https://arxiv.org/abs/2105.00003)
## NuSPAN: Nonuniform Sparse Proximal Averaging Network

### code for testing and training synthetic, simulated and real seismic reflection data.

The codes have been built on **PyTorch** (>=1.7) and **Python** (>=3.8), and tested on MacOS 10.13.6 and Ubuntu 18.04.

## Contents
1. [Setup](#setup)
2. [Test](#test)
3. [Train - NuSPAN](#train-NuSPAN)
4. [Figure for Real Data](#figure-for-real-data)
5. [Trained NuSPAN Models](#trained-nuspan-models)
6. [Function Definitions](#function-definitions)

## Setup

We recommend setting up a new **[conda](https://docs.conda.io/projects/conda/en/latest/)** environment (requires [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on the system). Create a new environment using the following command:
```
conda env create -f nuspan.yml
```

## Test

### Usage
To test any of the 4 datasets and reproduce results for our paper, simply run: 
```
python {a}_test.py {b}
```
#### {a} is the dataset [trace, wedge, marmousi2, real]

To test on either the **Simulated 2-D Marmousi2 Model** or the **Real Data**, please unzip the files *marmousi2_reflectivity.npy.zip* and *penobscot_3d.npy.zip*, respectively, before running the above commands.

The file *marmousi2_reflectivity.npy.zip* has been generated from the P-wave velocity and Density profiles provided for the Marmousi2 model.

Reference: https://wiki.seg.org/wiki/AGL_Elastic_Marmousi

To download, click the link or paste the line below into terminal: 
```
wget https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz
```


Link: https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz


The file *penobscot_3d.npy.zip* has been generated from the raw seismic data provided for the Penobscot 3D Survey. Reference: O. S. R., dGB Earth Sciences. Penobscot 3D - Survey, 2017). The raw data has been retrieved from https://terranubis.com/datainfo/Penobscot.

We also use the files *las.py* and *L-30.las* for our experiments on real data. Reference: Bianco, E. Geophysical tutorial: well-tie calculus. The Leading Edge, 33(6):674–677, 2014. Available in the original form at https://github.com/seg/tutorials-2014/tree/master/1406_Make_a_synthetic

#### {b} is the method/algorithm from the following list:

    -b, --bpi           (Basis-Pursuit Inversion - BPI)
    -f, --fista         (Fast Iterative Shrinkage-Thresholding Algorithm - FISTA)
    -s, -sblem          (Expectation-Maximization-based Sparse Bayesian Learning-  SBL-EM)
    -n1, --nuspan1      (Nonuniform Sparse Proximal Averaging Network Type 1 - NuSPAN-1)
    -n2, --nuspan2      (Nonuniform Sparse Proximal Averaging Network Type 2 - NuSPAN-2)

For help, run
```
python {a}_test.py -h
```
or
```
python {a}_test.py --help
```

### Examples:

#### 1. To generate results for **Synthetic 1-D Seismic Traces** using **NuSPAN-2**:
```
python trace_test.py -n2
```
or
```
python trace_test.py --nuspan2
```

#### 2. To generate results for **Synthetic 2-D Wedge Models** using **NuSPAN-1**:
```
python wedge_test.py -n1
```
or
```
python wedge_test.py --nuspan1
```

#### 3. To generate results for the **Simulated 2-D Marmousi Model** using **FISTA**:
```
python marmousi2_test.py -f
```
or
```
python marmousi2_test.py --fista
```

#### 4. To generate results for **Real Data** using **SBL-EM**:
```
python real_test.py -s
```
or
```
python real_test.py --sblem
```

## Train NuSPAN

To train **NuSPAN-1** or **NuSPAN-1**, follow a similar approach as above:

#### 1. Train **NuSPAN-1**:
```
python train nuspan -n1
```
or
```
python train nuspan --nuspan1
```

#### 2. Train **NuSPAN-2**:
```
python train nuspan -n2
```
or
```
python train nuspan --nuspan2
```

## Figure for Real Data

Run
```
python figure_real.py
```
to generate *Figure 5* in our paper, i.e., for results from a real dataset. This will generate the file *fig_real_xline_115.pdf*. The figure is generated using output arrays saved in the folders *real_bpi*, *real_fista*, *real_nuspan1*, *real_nuspan2*, and *real_sblem*.

## Trained NuSPAN Models

We provide both **NuSPAN-1** and **NuSPAN-2** trained models for all datasets. Different models are trained for different datasets mainly due to the variation of three model parameters, namely, the Ricker wavelet frequency, sampling interval, and the amplitude increment.

The models *nuspan1_trace_wedge_10.pt*, *nuspan1_trace_wedge_15.pt*, and *nuspan2_trace_wedge.pt* cater to the **Synthetic 1-D Traces** and **2-D Wedge Models**. Parameters of the seismic traces: 

    30 Hz Ricker wavelet frequency
    1 ms sampling interval
    0.2 amplitude increment

The models *nuspan1_marmousi2.pt* and *nuspan2_marmousi2.pt* cater to the **Simulated 2-D Marmousi2 Model**.

    30 Hz Ricker wavelet frequency
    2 ms sampling interval
    0.05 amplitude increment


The models *nuspan1_real.pt* and *nuspan2_real.pt* cater to the **Real Data**.

    25 Hz Ricker wavelet frequency
    4 ms sampling interval
    0.05 amplitude increment

## Function Definitions

The files *def_algorithms.py*, *def_figs.py*, and *def_models.py* contain the definitions of all the methods (**NuSPAN-1**, **NuSPAN-2**, BPI, FISTA, and SBL-EM), figures, and models, respectively.
