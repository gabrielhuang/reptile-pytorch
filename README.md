# Reptile

PyTorch implementation of OpenAI's Reptile algorithm for supervised learning.

Currently, it runs on Omniglot but not yet on MiniImagenet.

The code  has not been tested extensively. Contributions and feedback are more than welcome!

## Omniglot meta-learning dataset

There is already an Omniglot dataset class in torchvision, however it seems to be more adapted for supervised-learning
than few-shot learning.

The `omniglot.py` provides a way to sample K-shot N-way base-tasks from Omniglot, 
and various utilities to split meta-training sets as well as base-tasks.

## Features

- [x] Monitor training with TensorboardX.
- [x] Interrupt and resume training. 
- [x] Train and evaluate on Omniglot.
- [ ] Meta-batch size > 1.
- [ ] Train and evaluate on Mini-Imagenet.
- [ ] Clarify Transductive vs. Non-transductive setting.
- [ ] Add training curves in README.
- [ ] Reproduce all settings from OpenAI's code.
- [ ] Shell script to download datasets

## How to train on Omniglot

Download the two parts of the Omniglot dataset:
- https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
- https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip

Create a `omniglot/` folder in the repo, unzip and merge the two files to have the following folder structure:
```
./train_omniglot.py
...
./omniglot/Alphabet_of_the_Magi/
./omniglot/Angelic/
./omniglot/Anglo-Saxon_Futhorc/
...
./omniglot/ULOG/
```

Now start training with
```
python train_omniglot.py log --cuda 0 $HYPERPARAMETERS  # with CPU
python train_omniglot.py log $HYPERPARAMETERS  # with CUDA
```
where $HYPERPARAMETERS depends on your task and hyperparameters.

Behavior:
- If no checkpoints are found in `log/`, this will create a `log/` folder to store tensorboard information and checkpoints.
- If checkpoints are found in `log/`, this will resume from the last checkpoint.

Training can be interrupted at any time with `^C`, and resumed from the last checkpoint by re-running the same command.

## Omniglot Hyperparameters

The following set of hyperparameters work decently. 
They are taken from the OpenAI implementation but are adapted slightly
for `meta-batch=1`.

<img src="https://github.com/gabrielhuang/reptile-pytorch/raw/master/plots/omniglot_train.png" width="400">
<img src="https://github.com/gabrielhuang/reptile-pytorch/raw/master/plots/omniglot_val.png" width="400">

For 5-way 5-shot (red curve):

```bash
python train_omniglot.py log/o55 --classes 5 --shots 5 --train-shots 10 --meta-iterations 100000 --iterations 5 --test-iterations 50 --batch 10 --meta-lr 0.2 --lr 0.001
```

For 5-way 1-shot (blue curve):

```bash
python train_omniglot.py log/o51 --classes 5 --shots 1 --train-shots 12 --meta-iterations 200000 --iterations 12 --test-iterations 86 --batch 10 --meta-lr 0.33 --lr 0.00044
```




## References

- [Original Paper](https://arxiv.org/abs/1803.02999): Alex Nichol, Joshua Achiam, John Schulman. "On First-Order Meta-Learning Algorithms".
- [OpenAI blog post](https://blog.openai.com/reptile/). 
Check it out, they have an online demo running entirely in Javascript!
- Original code in Tensorflow: https://github.com/openai/supervised-reptile
