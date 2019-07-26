
import torch

from settings import *


class ConvNet3D(torch.nn.Module):
    """
    An fMRI slice given by BOLD5000 has shape: (71, 89, 72)
    """
    def __init__(self, num_outputs):
        super(ConvNet3D, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=8,
                               kernel_size=7, stride=2)
        self.conv2 = torch.nn.Conv3d(in_channels=8, out_channels=16,
                               kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv3d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=2)
        self.linear1 = torch.nn.Linear(14112, 128)
        self.linear2 = torch.nn.Linear(128, num_outputs)
        self.dropout1 = torch.nn.Dropout3d(p=DROPOUT_PROB)
        self.relu = torch.nn.ReLU()

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, x):

        # NO BATCH NORM // MAX POOL?

        _batch_size = x.shape[0]
        ### Convolution Layers
        # Current Size (batch_size, 1, 71, 89, 72)
        output = self.conv1(x)
        output = self.relu(output)
        # Current Size (batch_size, 16, 15, 19, 15)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.dropout1(output)
        # Current Size (batch_size, 32, 7, 9, 7)
        output = self.conv3(output)
        output = self.relu(output)

        output = output.reshape(_batch_size, -1)
        ### Fully Connected Layers
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)

        return output
