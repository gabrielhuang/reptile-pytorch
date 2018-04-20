# Reptile

My PyTorch implementation of OpenAI's Reptile algorithm for supervised learning.

Currently, it runs on Omniglot but not yet on MiniImagenet.

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

## References

- [Original Paper](https://arxiv.org/abs/1803.02999): Alex Nichol, Joshua Achiam, John Schulman. "On First-Order Meta-Learning Algorithms".
- [OpenAI blog post](https://blog.openai.com/reptile/). 
Check it out, they have an online demo running entirely in Javascript!
- Original code in Tensorflow: https://github.com/openai/supervised-reptile
