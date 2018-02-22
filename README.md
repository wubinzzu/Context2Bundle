# Context2Bundle
Context2Bundle: Diversified Personalized Bundle Recommendation

Context2Bundle(C2B) is a personalized bundle recommendation framework, which uses a customized sequence generation network to encode user context and generate candidate bundles, then controls the diversity of the candidates and forms the final bundle recommendation list by MWIS(maximum weighted k-induced subgraph) as re-ranking strategy.

This repo contains the experiment of Context2Bundle recommendation in the Amazon Dataset. We are glad to release the code in the Steam Dataset if necessary in the future. You are welcome to reproduce the experiment's result or adapt the code to your application.

## Requirements
* [Gurobi](http://www.gurobi.com/downloads/download-center) >= 7.5: Gurobi is a mathematical programming solver, we use it as one of the solvers of MWIS problem.
* [Python](https://www.anaconda.com/download/) >= 3.6 with numpy, scipy, sklearn, pandas, tqdm, gurobipy
* [TensorFlow](https://www.tensorflow.org/) >= 1.4.0: Probably earlier version should work too, though I didn't test it.
* GPU with memory >= 8G

## Download Dataset and Preprocess
* Clone this repo first, then download the Amazon Clothes Dataset.
```
cd data/
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
```
You can also download the Amazon Electro Dataset using `utils/download.sh`. For convenience, the following instructions only refer to Clothes Dataset.

* The preprocess script mainly parses the raw data, remaps all kinds of id, builds the bundle by co-purchase items and produces the training set and validation set. We regard the co-purchase items in the same timestamp as a bundle, put the first m-1 bundles as training set and the last one in the validation set. This script may spend a few minutes, so we use `tqdm` to show the progress bar for each preprocess stage.
```
$ cd ../utils/
$ python build.py
```
This script produces the output data we shall use later in `data/bundle_clo.pkl`, containing the training set, the validation set, the bundle map, the ground truth for generation measure, [etc](https://github.com/jinze1994/Context2Bundle/blob/b127cbeaeaacb7a280d02b3e00902276018d546f/utils/build.py#L130).

## Training the C2B Genaration Network
The code of C2B genaration network is located in `c2b/` directory. We train and evaluate the network with default arguments using the preprocessed data in `data/bundle_clo.pkl`. We print the log to file and run in background mode. 
```
$ cd ../c2b/
$ python model.py >tra.log 2>&1 &
```
In `tra.log` we would see
```
...
Epoch:13 AUC:0.6994
model saved at clo/model-20826
...
```
which indicates that we reach the `69.94%` in AUC for ranking in the validation set. Note that the AUC is not the measure of generation but the measure of ranking. During training we have saved the checkpoint of model in `c2b/clo/` directory, so we could restore all network weights and generate candidate bundles by running
```
$ python gen.py
```
This script would restore the checkpoint in `clo/model-20826`, generate 50 candidate bundles and save them to `res_clo_50.pkl`. We compare the generative results with the ground truth and calculate the precision and diversity subsequently, we could see in script's output
```
100%|███████████████████| 1000/1000 [00:28<00:00, 35.31it/s]
clo     P@10: 0.3553%   Div: 0.2147
```
which indicates the precision and diversity for generation. And we could generate bundles list for `35` users per second, the batch size is set to one here.

## Bundle Re-ranking
We could notice that the C2B is inclined to generate very similar bundles for reaching higher precision without controlling. So we introduce the problem of finding maximum weighted k-induced subgraph as the re-ranking strategy. The code of C2B re-ranking is located in `miqp/` directory.

The script `MIQP.py` could select k bundles as final results from the candidate bundles `res_clo_50.pkl`. There are three mode for this script, `basic`, `greedy` and `gurobi`, where `basic` is the origin `C2B`, `greedy` use the greedy algorithm as we state in paper, and `gurobi` use the programming solver in Gurobi MIQP tools.
```
$ cd ../miqp
$ python MIQP.py ../c2b/res_clo_50.pkl
100%|███████████████████| 1000/1000 [00:03<00:00, 287.65it/s]
../c2b/res_clo_50.pkl   P@10: 0.2730%   Div: 0.0685
```
which indicates the precision and diversity after the bundle re-ranking.
The script uses `greedy` mode by default. You can change it [here](https://github.com/jinze1994/Context2Bundle/blob/b127cbeaeaacb7a280d02b3e00902276018d546f/miqp/MIQP.py#L12). You must get right licence first and install the [Gurobi](http://www.gurobi.com/downloads/download-center) if using `gurobi` mode.
We could notice that we get better diversity with acceptable decline of presicion.
