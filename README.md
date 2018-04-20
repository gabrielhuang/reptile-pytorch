# Reptile

My PyTorch implementation of OpenAI's Reptile algorithm for supervised learning.

Currently, it runs on Omniglot but not yet on MiniImagenet.

I have not tested the code extensively. Contributions and feedback are more than welcome!

## Omniglot meta-learning dataset

I know there is already an Omniglot dataset class in torchvision, however it seems to be more adapted for supervised-learning
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

## References

- [Original Paper](https://arxiv.org/abs/1803.02999): Alex Nichol, Joshua Achiam, John Schulman. "On First-Order Meta-Learning Algorithms".
- [OpenAI blog post](https://blog.openai.com/reptile/). 
Check it out, they have an online demo running entirely in Javascript!
- Original code in Tensorflow: https://github.com/openai/supervised-reptile
