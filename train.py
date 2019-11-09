
import time
from argparse import ArgumentParser

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

from model import ConvNet3D, ConvNet3D_Old
from data import BOLD5000
from settings import *


def iterate(loader):
    losses = []
    predictions = []
    targets = []
    for i, (images, labels) in enumerate(loader):

        images = images.reshape(-1, 1, *MIN_3D_SHAPE).type(torch.FloatTensor)
        if model.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # Backprop and perform Adam optimisation
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Track the accuracy
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    return (np.mean(losses), accuracy_score(targets, predictions),
            f1_score(targets, predictions, average='micro'),
            confusion_matrix(targets, predictions, dataset.classes))


parser = ArgumentParser()
parser.add_argument('--level', default='primary',
                    help='primary, secondary, or tertiary')
parser.add_argument('--save-model', default=False)
parser.add_argument('--num-epochs', default=str(NUM_EPOCHS))
parser.add_argument('--batch-size', default=str(BATCH_SIZE))
parser.add_argument('--lr', default=str(INITIAL_LR))
args = parser.parse_args()

batch_size = eval(args.batch_size)
num_epochs = eval(args.num_epochs)

dataset = BOLD5000(args.level)
model = ConvNet3D(num_outputs=dataset.num_classes)
print('----- Model -----\n', model)

class_weights = torch.Tensor(
    compute_class_weight('balanced', dataset.classes, dataset.labels))

criterion = torch.nn.CrossEntropyLoss(class_weights.cuda())
optimizer = torch.optim.Adam(model.parameters(), lr=eval(args.lr))

train_loader, test_loader = dataset.train_test_split(batch_size)

# Train the model
# total_step = len(train_loader)
# loss_list = []
# acc_list = []

print('\n----- PARAMETERS -----')
print('Batch Size:', batch_size)
print('Train Batches:', len(train_loader), '\nTest Batches:', len(test_loader))
print('Initial Learning Rate:', args.lr)
print('Color Classification Level:', args.level)
print('Loss Class Weights:', class_weights)

print('\n----- Training for {} epochs -----'.format(num_epochs))
print('Start time:', time.strftime('%H:%M'))

for epoch in range(num_epochs):
    model.train()
    train_loss, train_acc, train_f1, train_cm = iterate(train_loader)
    model.eval()
    test_loss, test_acc, test_f1, test_cm = iterate(test_loader)

    print('\n[{}] [{}/{}] | '.format(time.strftime('%H:%M'), epoch + 1, num_epochs),
          'Train Loss: {:.4f}, Acc: {:.2f}% F1: {:.2f} |'.format(train_loss, train_acc * 100, train_f1),
          'Test Loss: {:.4f}, Acc: {:.2f}% F1 {:.2f}'.format(test_loss, test_acc * 100, test_f1))
    print('Train Confusion Matrix:\n', train_cm)
    print('Test Confusion Matrix:\n', test_cm)
