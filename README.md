# cpl

This repository contains the implementation used in our CP23 paper. The implementation aims at generating decision sets that are both interpretable and accurate, by compiling a gradient boosted tree model on demand, where each generated rule is equivalent to an abductive explanation for the prediction made by the gradient boosted tree. The experiments compare the proposed implementation with other state-of-the-art decision set learning algorithms in terms of accuracy, scalability, model size and explanation size.

## Instruction <a name="instrt"></a>
Before using the implementation, we need to extract the datasets stored in ```datasets.tar.xz```. To extract the datasets, please ensure ```tar``` is installed and run:
```
$ tar -xvf datasets.tar.xz
```

If interested in the logs, please run:
```
$ tar -xvf logs.tar.xz
```

## Table of Content
* **[Required Packages](#require)**
* **[Usage](#usage)**
	* [Prepare a dataset](#prepare)
	* [Generate a boosted tree](#bt)
	* [Compile a boosted tree into a decision set](#cpl)
* **[Reproducing Experimental Results](#expr)**

## Required Packages <a name="require"></a>
The implementation is written as a set of Python scripts. The python version used in the experiments is 3.8.5. Some packages are required. To install requirements:
```
$ pip install -r requirements.txt
```

In addition to the packages above, Gurobi with full licence is also required. To install Gurobi, please follow the [instruction](https://www.gurobi.com/). Please also follow the [instruction](https://github.com/jirifilip/pyIDS/) to install IDS.

## Usage <a name="usage"></a>
`cpl.py` provides a number of parameters, which can be set from the command line. To see the list of parameters, run:
```
$ cd src/ && python cpl.py -h
```

### Preparing a dataset <a name="prepare"></a>  <a name="prepare"></a>
`Cpl` can address datasets in the CSV format. Before compiling a gradient boosted tree (BT) model in to a decision set (DS), we need to prepare the datasets the train a BT model.

1. Assume a target dataset is stored in ```somepath/dataset.csv```
2. Create an extra file named ```somepath/dataset.csv.catcol``` containing the indices of the categorical columns ofthe target dataset. For example, if columns ```0```, ```3```, and ```6``` are categorical features, the file should be as follow:
	```
	0
	3
	6
	```
3. With the two files above, we can run:
```
$ python cpl.py -p --pfiles dataset.csv,somename somepath/
```
to create a new dataset file `somepath/somename_data.csv` with the categorical features properly addressed. For example:
```
$ python cpl.py -p --pfiles iris_train1.csv,iris_train1 ../datasets/train/iris/
```

### Training a gradient boosted tree model  <a name="bt"></a>
A gradient boosted tree model is required before generating a decision set. Run the following command to train a BT model:
```
$ python cpl.py -c -t -n 50 -d 3 --testsplit 0 ../datasets/train/iris/iris_train1_data.csv 
```
Here, a boosted tree consisting of 50 trees per class is trained, where the maximum depth of each tree is 3. ``` ../datasets/train/iris/iris_train1_data.csv ``` is the dataset to be trained. The value of ```--testsplit``` ranges from 0.0 to 1.0. In this command line, the given dataset is split into 100% to train and 0% to test. By default, the generated model is saved in ```./temp/iris_train1_data/iris_train1_data_nbestim_50_maxdepth_3_testsplit_0.0.mod.pkl```

### Compiling a boosted tree into a decision set  <a name="cpl"></a>
To generate a decision set via local compilation, i.e. the computed decision set covers all instances in the training dataset:
```
$ python cpl.py -f -I -R lin -e mx -s g3 -v --clocal --fsort --fqupdate ./temp/iris_train1_data/iris_train1_data_nbestim_50_maxdepth_3_testsplit_0.0.mod.pkl

```
```-f``` enables the compiled decision set in a particular format. ```-I -R lin``` activates the compilation process where the standard linear search for rule extraction is used. ```-e mx -s g3``` indicates the MaxSAT encoding and g3 SAT solver are used. ```-v``` increases verbosity level. ``` --clocal --fsort --fqupdate``` indicates local compilation and the feature sorting based on feature frequencies is activated.

Lexicographic optimization on each rule, i.e. minimizing misclassifications first then the number of literals used,  can be activated by adding ```--reduce-lit after --reduce-lit-appr maxsat```.
```
$ python cpl.py -f -I -R lin -e mx -s g3 -v --clocal --fsort --fqupdate --reduce-lit after --reduce-lit-appr maxsat ./temp/iris_train1_data/iris_train1_data_nbestim_50_maxdepth_3_testsplit_0.0.mod.pkl
```

To enable the tradeoff between misclassifications and the number of literals used in each rule, add ```--lam 0.005 --approx 1 ```:
```
$ python cpl.py -f -I -R lin -e mx -s g3 -v --clocal --fsort --fqupdate --reduce-lit after --reduce-lit-appr maxsat --lam 0.005 --approx 1 ./temp/iris_train1_data/iris_train1_data_nbestim_50_maxdepth_3_testsplit_0.0.mod.pkl
```

To activate rule reduction, add ```--reduce-rule --weighted```:
```
$ python cpl.py -f -I -R lin -e mx -s g3 -v --clocal --fsort --fqupdate --reduce-rule --weighted ./temp/iris_train1_data/iris_train1_data_nbestim_50_maxdepth_3_testsplit_0.0.mod.pkl
```

To activate both lexicographic optimization and rule reduction, add both ```` --reduce-lit after --reduce-lit-appr maxsat`` and ```--reduce-rule --weighted ```:
```
$ python cpl.py -f -I -R lin -e mx -s g3 -v --clocal --fsort --fqupdate --reduce-lit after --reduce-lit-appr maxsat --reduce-rule --weighted ./temp/iris_train1_data/iris_train1_data_nbestim_50_maxdepth_3_testsplit_0.0.mod.pkl
```

The implementation also supports exhaustive compilation:
```
$ python cpl.py -f -I -R lin -e mx -s g3 -v ./temp/iris_train1_data/iris_train1_data_nbestim_50_maxdepth_3_testsplit_0.0.mod.pkl
```

## Reproducing  Experimental Results <a name="expr"></a>
Due to randomization used in the training phase, it seems unlikely that the experimental results reported in the report can be completely reproduced.
Similar experimental results can be obtained by the following script:

```
$ ./src/experiment/repro_exp.sh
```

Since the total number of datasets is 295 and 13 decision set competitors are considered, running the experiments will take a while.
