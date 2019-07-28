
import torch

from settings import *


class ConvNet3D(torch.nn.Module):
    """
    An fMRI slice given by BOLD5000 has shape: (71, 89, 72)
    """
    def __init__(self, num_outputs):
        super(ConvNet3D, self).__init__()
        """
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=8,
                                     kernel_size=7, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(num_features=8)
        self.conv2 = torch.nn.Conv3d(in_channels=8, out_channels=16,
                                     kernel_size=5, stride=2)
        self.bn2 = torch.nn.BatchNorm3d(num_features=16)
        self.conv3 = torch.nn.Conv3d(in_channels=16, out_channels=32,
                                     kernel_size=3, stride=2)
        self.bn3 = torch.nn.BatchNorm3d(num_features=32)
        self.linear1 = torch.nn.Linear(14112, 128)
        self.linear2 = torch.nn.Linear(128, num_outputs)
        self.dropout1 = torch.nn.Dropout3d(p=DROPOUT_PROB)
        self.relu = torch.nn.ELU()
        """

        # Inspired by Qureshi et al. // VGG16
        # theirs is still bigger than this. Can try to increase features later.
        _ch_1 = 32
        _ch_2 = 64
        _ch_3 = 128
        _fc = 4096
        self.conv1_1 = torch.nn.Conv3d(in_channels=1, out_channels=_ch_1,
                                       kernel_size=3, stride=1)
        self.conv1_2 = torch.nn.Conv3d(in_channels=_ch_1, out_channels=_ch_1,
                                       kernel_size=3, stride=1)
        self.bn1 = torch.nn.BatchNorm3d(num_features=_ch_1)

        self.conv2_1 = torch.nn.Conv3d(in_channels=_ch_1, out_channels=_ch_2,
                                       kernel_size=3, stride=1)
        self.conv2_2 = torch.nn.Conv3d(in_channels=_ch_2, out_channels=_ch_2,
                                       kernel_size=3, stride=1)
        self.bn2 = torch.nn.BatchNorm3d(num_features=_ch_2)

        self.conv3_1 = torch.nn.Conv3d(in_channels=_ch_2, out_channels=_ch_3,
                                       kernel_size=3, stride=1)
        self.conv3_2 = torch.nn.Conv3d(in_channels=_ch_3, out_channels=_ch_3,
                                       kernel_size=3, stride=1)
        self.conv3_3 = torch.nn.Conv3d(in_channels=_ch_3, out_channels=_ch_3,
                                       kernel_size=3, stride=1)
        self.bn3 = torch.nn.BatchNorm3d(num_features=_ch_3)

        # self.conv4_1 = torch.nn.Conv3d(in_channels=64, out_channels=128,
        #                                kernel_size=3, stride=1)
        # self.conv4_2 = torch.nn.Conv3d(in_channels=128, out_channels=128,
        #                                kernel_size=3, stride=1)
        # self.conv4_3 = torch.nn.Conv3d(in_channels=128, out_channels=128,
        #                                kernel_size=3, stride=1)
        # self.bn4 = torch.nn.BatchNorm3d(num_features=128)

        self.fc1 = torch.nn.Linear(12288, _fc)
        self.fc2 = torch.nn.Linear(_fc, _fc)
        self.output = torch.nn.Linear(_fc, num_outputs)

        self.maxpool = torch.nn.MaxPool3d(kernel_size=2)
        self.dropout = torch.nn.Dropout3d(p=DROPOUT_PROB)
        self.activation = torch.nn.ReLU()

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, x):
        _batch_size = x.shape[0]

        # (B, 16, 34, 43, 35)
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.activation(self.bn1(out))
        # out = self.bn1(self.activation(out))
        out = self.maxpool(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.activation(self.bn2(out))
        # out = self.bn2(self.activation(out))
        out = self.maxpool(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.activation(self.bn3(out))
        # out = self.bn3(self.activation(out))
        out = self.maxpool(out)

        # out = self.conv4_1(out)
        # out = self.conv4_2(out)
        # out = self.conv4_3(out)
        # out = self.activation(self.bn4(out))
        # out = self.bn4(self.activation(out))
        # out = self.maxpool(out)

        out = out.reshape(_batch_size, -1)
        out = self.activation(self.dropout(self.fc1(out)))
        out = self.activation(self.dropout(self.fc2(out)))
        out = self.output(out)
        return out

        """
        ### Convolution Layers
        # Current Size (batch_size, 1, 71, 89, 72)
        output = self.conv1(x)
        output = self.relu(output)

        # Current Size (batch_size, 16, 15, 19, 15)
        output = self.conv2(output)
        output = self.relu(output)

        # Current Size (batch_size, 32, 7, 9, 7)
        output = self.conv3(output)
        output = self.relu(output)
        output = self.dropout1(output)
        output = output.reshape(_batch_size, -1)

        ### Fully Connected Layers
        output = self.linear1(output)
        output = self.dropout1(output)
        output = self.relu(output)
        output = self.linear2(output)

        return output
        """
