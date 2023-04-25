# LREANNtf

### Author

Richard Bruce Baxter - Copyright (c) 2020-2023 Baxter AI (baxterai.com)

### Description

Learning Rule Experiment artificial neural network (LREANN) for TensorFlow - experimental

expAUANN - associative (wrt exemplar) update
expCUANN - common (activation path) update
expHUANN - hebbian update
expMUANN - multi propagation (per layer; with synaptic delta calculation) update
expNUANN - (neuron activation) normalisation update
expRUANN - relaxation update
expSUANN - stochastic update
expXUANN - contrastive (pos/neg sample diff) update

### License

MIT License

### Installation
```
conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
conda install scikit-learn (ANNtf_algorithmLIANN_math:SVD/PCA only)
```

### Execution
```
source activate anntf2
python3 LREANNtf_main.py
```
