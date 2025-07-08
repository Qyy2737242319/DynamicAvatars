<p align="center">
    <a href="https://github.com/Qyy2737242319/xjtuse-intern-portrait/edit/main/rating_evaluate">
      <img src="https://img.shields.io/badge/Neural--ALS-1.0.0-brightgreen" alt="neural-ALS">
    </a>
    <a href="#">
        <img src="https://img.shields.io/github/license/huccct/vue-admin">
    </a>
</p>

This repository demonstrates our work, DynamicAvatars. Based on GaussianAvatars and GaussianEditor, DynamicAvatars has the ability to edit a dynamic 3D Gaussian Splatting facial avatars by text prompt.


## Experiment Results

<img src="./train_results/net_loss.png" width=auto height=325> </img>

<img src="./train_results/accuracy.png" width=auto height=250> </img>

## Installation

Clone our repository and create a conda environment to begin:
```

```
where test_data value should be replaced by batches of user_id and movies_id you interested in.


## Environment Preparation

Creating environment using bash command below:

```
conda env create -f requirements.yml
pip install -r requirements.txt
```

## Device Requirements & Training Details

* We deployed parallel computation on 3 nvidia rtx4090 gpus for training and evaluating stages and 20 cpu cores for data loading. Model parameters we used had set in the scripts.
* If you don't meet the requirements, you need to revise the code and some super parameters to enjoy our work.
* In our experiments, Neural-ALS needs about 9hrs to reduce the loss to 0.1 and 15hrs to reach convergence.
* During our experiments, we found that the super parameter k is proportional to the final accuracy, so increase the value of k as large as you can to enjoy the maximum performance.
* We recommand lr to be 1e-5 during the first 4000 iter and 5*1e-6 for the rest part of training due to numerical stability of the loss function.

## Training Settings

Train a Neural-ALS model via:

```
python ./recommand_predict.py --mode train [optional: --k --batch_size --epoch --lr --log_iter --save_iter --resume]
```
where super parameters could be passed by appendix 

## Additional Information
we provide some useful tools for real time monitoring.

### Use tensorboard to check training data

```
tensorboard --logdir ./logs
```
### Use matlpotlib to check training data

```
python ./tools/matloss.py ./logs
```

### Model Structure

<img src="./train_results/model_structure.png" width=auto height=800> </img>

### Parallel Computation Ability

<img src="./train_results/tpu_capability.png" width=auto height=300> </img>