# Context2Bundle
Context2Bundle: Diversified Personalized Bundle Recommendation

Context2Bundle(C2B) is a personalized bundle recommendation framework, which uses a customized sequence generation network to encode user context and generate candidate bundles, then controls the diversity of the candidates and forms the final bundle recommendation list by MWIS(maximum weighted k-induced subgraph) as re-ranking strategy.

This repo contains the experiment of Context2Bundle recommendation in the Amazon Dataset. For convenience, the following instructions only refer to Amazon Clothes Dataset, we are glad to release the code in the Steam Dataset if necessary in the future. You are welcome to reproduce the experiment's result or adapt the code to your application.

## Requirements
* [Gurobi](http://www.gurobi.com/downloads/download-center) >= 7.5: Gurobi is a mathematical programming solver, we use it as one of the solvers of MWIS problem.
* [Python](https://www.anaconda.com/download/) >= 3.6 with numpy, scipy, sklearn, pandas, tqdm, gurobipy
* [TensorFlow](https://www.tensorflow.org/) >= 1.4.0
* GPU with memory >= 8G

## Download Dataset and Preprocess
* Clone this repo first, then download the Amazon Clothes Dataset.
```
cd data/
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json
gzip -d reviews_Clothing_Shoes_and_Jewelry_5.json
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json
gzip -d meta_Clothing_Shoes_and_Jewelry.json
```
You can also download the Amazon Electro Dataset using the script `utils/download.sh`. For convenience, the following instructions only refer to Clothes Dataset.

* The preprocess script mainly parses the raw data, remaps all kinds of id, builds the bundle by co-purchase items and produces the training set and validation set. We regard the co-purchase items in the same timestamp as a bundle, put the first m-1 bundles as training set and the last one in the validation set. This script may spend a few minutes, so we use `tqdm` to show the progress bar for each stage.
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
The training process may spend about one hour, so we have saved the ckeckpoint and log file in this repo. In `tra.log` we would see
```
...
Epoch:12 AUC:0.6999
model saved at clo/model-19224
...
```
which indicates that we reach the `69.99%` in AUC for ranking in the validation set. Note that the AUC is not the measure of generation but the measure of ranking. During training we have saved the checkpoint of model in `c2b/clo/` directory, so we could restore all network weights and generate candidate bundles by running
```
$ python gen.py
```
This script would restore the checkpoint, generate 50 candidate bundles and save them to `res_clo_50.pkl`. We compare the generative results with the ground truth and calculate the precision and diversity subsequently, we could see in script's output
```
100%|███████████████████| 1000/1000 [00:28<00:00, 35.31it/s]
clo     P@10: 0.3865%   Div: -0.1837
```
which indicates the precision and diversity for generation. And the process bar shows that we could generate the bundles for `35` users per second, the batch size is set to one here.

## Bundle Re-ranking
We could notice that the C2B is inclined to generate very similar bundles and ignore diversity for reaching higher precision without controlling. So we introduce the problem of finding maximum weighted k-induced subgraph as the re-ranking strategy. The code of C2B re-ranking is located in `mwis/` directory.

The script `MIQP.py` could select k bundles as final results from the candidate bundles `res_clo_50.pkl`. There are two extra mode for this script, `greedy` and `gurobi`, where `greedy` use the greedy algorithm as we state in paper, and `gurobi` use the programming solver in Gurobi's Mixed Integer Quadratic Programming(MIQP) tools.
```
$ cd ../mwis
$ python MWIS.py ../c2b/res_clo_50.pkl
100%|███████████████████| 1000/1000 [00:03<00:00, 287.65it/s]
../c2b/res_clo_50.pkl   P@10: 0.2559%   Div: -0.0542
```
which indicates the precision and diversity after the bundle re-ranking.
The script uses `greedy` mode by default. You can change it [here](https://github.com/jinze1994/Context2Bundle/blob/b127cbeaeaacb7a280d02b3e00902276018d546f/miqp/MIQP.py#L12). You must get right licence first and install the [Gurobi](http://www.gurobi.com/downloads/download-center) if using `gurobi` mode.
We could notice that we get better diversity with acceptable decline of presicion.

## Competitors
Here we show the competitors' implementation.

### Freq

We count the frequency directly of each subset of the co-purchase bundles, and recommend the most frequent ones as generation bundles without personalization.
```
$ cd ../top/
$ python top_gen.py
clo P@5: 0.0828%        Div: -0.0333
clo P@10: 0.0955%       Div: -0.0593
```

### Bundle-rank

We could use cnn network to extract both the user and the bundle context, and use a pairwise loss function like BPR. The best AUC in validation set using this method is `66.31%`.
```
$ cd ../rank/
$ python cnn.py
...
Epoch:5 AUC:0.6631
model saved at clo/model-16020
...
```

Furthermore, we could use a ranking model to generate bundle for user, by ranking all co-purchase bundles which has appeared. For the sake of fairness, we use our best bundle ranking model.
```
$ cd ../c2b/
$ python rank.py >rank.log 2>&1 &
```
The script `rank.py` could rank all co-purchase bundles which has appeared, and save all logits to file `rank_clo_all.pkl`. This script may spend a few hours. Then we could calculate the precision and diversity for generation.
```
$ python rank_res.py
clo     P@10: 0.2803%   Div: 0.1077
```
