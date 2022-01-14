# A New Perspective on the Effects of Spectrum in Graph Neural Networks

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This is the code of the paper
*A New Perspective on the Effects of Spectrum in Graph Neural Networks*.

## Table of Contents <!-- omit in toc -->

- [Requirements](#requirements)
- [Run Experiments](#run-experiments)
  - [ZINC](#zinc)
  - [OGBG-MolPCBA](#ogbg-molpcba)
  - [TU](#tu)
- [License](#license)

## Requirements

The code is built upon [Deep Graph Library](https://www.dgl.ai/).

The following packages need to be installed:

- `torch==1.8.0`
- `dgl==0.7.0`
- `torch-geometric==1.7.1`
- `ogb==1.3.1`
- `numpy`
- `easydict`
- `tensorboard`

## Run Experiments

### ZINC

To run experiments on `ZINC`, change directory to [zinc](zinc):

```sh
cd zinc
```

You can set hyper-parameters in
[ZINC.json](zinc/ZINC.json).

Then, run the following script:

```sh
./run_script.sh
```

### OGBG-MolPCBA

To run experiments on `ogbg-mol*`, change directory to [ogbg/mol](ogbg/mol):

```sh
cd ogbg/mol
```

You can set hyper-parameters in
[ogbg-molpcba.json](ogbg/mol/ogbg-molpcba.json).

Then, run the following script:

```sh
./run_script.sh
```

### TU

To run experiments on `TU`, change directory to [tu](tu):

```sh
cd tu
```

There are several datesets in TU.
You can set dataset name in [run_script.sh](tu/run_script.sh)
and set hyper-parameters in [configs/\<dataset\>.json](tu/configs).

Then, run the following script:

```sh
./run_script.sh
```

## License

[MIT License](LICENSE)
