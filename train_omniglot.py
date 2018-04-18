import argparse
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from models import OmniglotModel
from omniglot import MetaOmniglotFolder, split_omniglot, ImageCache, transform_image, transform_label


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

def Variable_(tensor, *args_, **kwargs):
    '''
    Make variable cuda depending on the arguments
    '''
    # Unroll list or tuple
    if type(tensor) in (list, tuple):
        return [Variable_(t, *args_, **kwargs) for t in tensor]
    # Unroll dictionary
    if isinstance(tensor, dict):
        return {key: Variable_(v, *args_, **kwargs) for key, v in tensor.items()}
    # Normal tensor
    variable = Variable(tensor, *args_, **kwargs)
    if args.cuda:
        variable = variable.cuda()
    return variable

# Parsing
parser = argparse.ArgumentParser('Train reptile on omniglot')

# - Training params
parser.add_argument('--classes', default=5, type=int, help='classes in base-task (N-way)')
parser.add_argument('--shots', default=1, type=int, help='shots per class (K-shot)')
parser.add_argument('--meta-iterations', default=400000, type=int, help='number of meta iterations')
parser.add_argument('--base-iterations', default=400000, type=int, help='number of meta iterations')
parser.add_argument('--base-batch', default=8, type=int, help='minibatch size in base task')

# - General params
parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
parser.add_argument('--validate-every', default=10, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--input', default='omniglot', help='Path to omniglot dataset')
parser.add_argument('--output', help='Where to save models')
parser.add_argument('--cuda', default=1, type=int, help='Use cuda')
args = parser.parse_args()

# Load data
# Resize is done by the MetaDataset because the result can be easily cached
omniglot = MetaOmniglotFolder(args.input, size=(28, 28), cache=ImageCache(),
                              transform_image=transform_image,
                              transform_label=transform_label)
train_dataset, test_dataset = split_omniglot(omniglot, args.validation)


print 'Meta-Train characters', len(train_dataset)
print 'Meta-Test characters', len(test_dataset)

# Load model
meta_net = OmniglotModel(args.classes)
if args.cuda:
    meta_net.cuda()

# Loss
cross_entropy = nn.CrossEntropyLoss()
def get_loss(prediction, labels):
    return cross_entropy(prediction, labels)


def do_base_learning(base_net, base_optimizer, base_iter, base_iterations):

    for base_iteration in xrange(base_iterations):
        # Sample minibatch
        data, labels = Variable_(base_iter.next())

        # Forward pass
        prediction = base_net(data)

        # Get loss
        loss = get_loss(prediction, labels)

        # Backward pass - Update fast net
        base_optimizer.zero_grad()
        loss.backward()
        base_optimizer.step()

    return loss.data[0]


def do_base_evaluation(base_net, base_iter, base_iterations):

    losses = []
    for base_iteration in xrange(base_iterations):
        # Sample minibatch
        data, labels = Variable_(base_iter.next())

        # Forward pass
        prediction = base_net(data)

        # Get loss
        loss = get_loss(prediction, labels)

        losses.append(loss.data[0])

    return np.mean(losses)


# Main loop
meta_optimizer = torch.optim.Adam(meta_net.parameters())
for meta_iteration in tqdm(xrange(args.meta_iterations)):

    # Clone model
    base_net = meta_net.clone()
    base_optimizer = torch.optim.Adam(base_net.parameters())
    # load state of base optimizer?

    # Sample base task
    base_task = train_dataset.get_random_task(args.classes, args.shots)
    base_iter = make_infinite(DataLoader(base_task, args.base_batch, shuffle=True))

    # Update fast net
    loss = do_base_learning(base_net, base_optimizer, base_iter, args.base_iterations)

    # Update slow net
    meta_net.point_grad_to(base_net)
    meta_optimizer.step()

    # Meta-Evaluation
    if meta_iteration % args.validate_every == 0:
        for (meta_dataset, mode) in [(train_dataset, 'train'), (test_dataset, 'val')]:

            base_train_task, base_test_task = meta_dataset.get_random_task_split(args.classes, train_K=args.shots, test_K=1)
            base_train_iter = make_infinite(DataLoader(base_train_task, args.base_batch, shuffle=True))
            base_test_iter = make_infinite(DataLoader(base_train_task, args.base_batch, shuffle=True))

            # Base-train
            base_net = meta_net.clone()
            loss = do_base_learning(base_net, base_optimizer, base_train_iter, args.base_iterations)

            # Base-test: compute meta-loss, which is base-validation error
            meta_loss = do_base_evaluation(base_net, base_test_iter)

            print '{} metaloss', meta_loss
