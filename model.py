
import torch

from settings import *


class ConvNet3D(torch.nn.Module):
    """
    An fMRI slice given by BOLD5000 has shape: (71, 89, 72)
    """
    def __init__(self, num_outputs):
        super(ConvNet3D, self).__init__()
        # Inspired by Qureshi et al. // VGG16
        # theirs is still bigger than this. Can try to increase features later.

        # 32, 64, 128, 256 = overfits!
        _ch_1 = 16
        _ch_2 = 32
        _ch_3 = 64
        _ch_4 = 512
        _fc = 2048

        # Conv Block 1
        self.conv1_1 = torch.nn.Conv3d(in_channels=1, out_channels=_ch_1,
                                       kernel_size=3, stride=1)
        self.conv1_2 = torch.nn.Conv3d(in_channels=_ch_1, out_channels=_ch_1,
                                       kernel_size=3, stride=1)
        self.bn1_1 = torch.nn.BatchNorm3d(num_features=_ch_1)
        self.bn1_2 = torch.nn.BatchNorm3d(num_features=_ch_1)

        # Conv Block 2
        self.conv2_1 = torch.nn.Conv3d(in_channels=_ch_1, out_channels=_ch_2,
                                       kernel_size=3, stride=1)
        self.conv2_2 = torch.nn.Conv3d(in_channels=_ch_2, out_channels=_ch_2,
                                       kernel_size=3, stride=1)
        self.conv2_3 = torch.nn.Conv3d(in_channels=_ch_2, out_channels=_ch_2,
                                       kernel_size=3, stride=1)
        self.bn2_1 = torch.nn.BatchNorm3d(num_features=_ch_2)
        self.bn2_2 = torch.nn.BatchNorm3d(num_features=_ch_2)
        # self.bn2_3 = torch.nn.BatchNorm3d(num_features=_ch_2)

        # Conv Block 3
        self.conv3_1 = torch.nn.Conv3d(in_channels=_ch_2, out_channels=_ch_3,
                                       kernel_size=3, stride=1)
        self.conv3_2 = torch.nn.Conv3d(in_channels=_ch_3, out_channels=_ch_3,
                                       kernel_size=3, stride=1)
        self.conv3_3 = torch.nn.Conv3d(in_channels=_ch_3, out_channels=_ch_3,
                                       kernel_size=3, stride=1)
        # self.conv3_4 = torch.nn.Conv3d(in_channels=_ch_3, out_channels=_ch_3,
        #                                kernel_size=3, stride=1)
        self.bn3_1 = torch.nn.BatchNorm3d(num_features=_ch_3)
        self.bn3_2 = torch.nn.BatchNorm3d(num_features=_ch_3)
        self.bn3_3 = torch.nn.BatchNorm3d(num_features=_ch_3)
        # self.bn3_4 = torch.nn.BatchNorm3d(num_features=_ch_3)

        # Conv Block 4
        # self.conv4_1 = torch.nn.Conv3d(in_channels=_ch_3, out_channels=_ch_4,
        #                                kernel_size=3, stride=1)
        # self.conv4_2 = torch.nn.Conv3d(in_channels=_ch_4, out_channels=_ch_4,
        #                                kernel_size=3, stride=1)
        # self.conv4_3 = torch.nn.Conv3d(in_channels=_ch_4, out_channels=_ch_4,
        #                                kernel_size=3, stride=1)
        # self.bn4_1 = torch.nn.BatchNorm3d(num_features=_ch_4)
        # self.bn4_2 = torch.nn.BatchNorm3d(num_features=_ch_4)
        # self.bn4_3 = torch.nn.BatchNorm3d(num_features=_ch_4)

        self.fc1 = torch.nn.Linear(4608, _fc)
        # self.fc2 = torch.nn.Linear(_fc, _fc)
        self.output = torch.nn.Linear(_fc, num_outputs)

        self.maxpool = torch.nn.MaxPool3d(kernel_size=2)
        self.dropout = torch.nn.Dropout3d(p=DROPOUT_PROB)
        self.activation = torch.nn.ReLU()

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, x):
        def _activate(out, bn):
            return bn(self.activation(out))
            # or:
            # return self.activation(bn(out))

        _batch_size = x.shape[0]

        # (B, 16, 34, 43, 35)
        conv1_1 = self.conv1_1(x)
        conv1_1_act = _activate(conv1_1, self.bn1_1)  # Qureshi did not bn+act after first conv
        conv1_2 = self.conv1_2(conv1_1_act)
        conv1_2_act = _activate(conv1_2, self.bn1_2)
        conv1 = self.maxpool(conv1_2_act)

        conv2_1 = self.conv2_1(conv1)
        conv2_1_act = _activate(conv2_1, self.bn2_1)
        conv2_2 = self.conv2_2(conv2_1_act)
        conv2_2_act = _activate(conv2_2, self.bn2_2)
        # out = self.conv2_3(out)
        # out = _activate(out, self.bn2_3)
        conv2 = self.maxpool(conv2_2_act)

        conv3_1 = self.conv3_1(conv2)
        conv3_1_act = _activate(conv3_1, self.bn3_1)
        conv3_2 = self.conv3_2(conv3_1_act)
        conv3_2_act = _activate(conv3_2, self.bn3_2)
        conv3_3 = self.conv3_3(conv3_2_act)
        conv3_3_act = _activate(conv3_3, self.bn3_3)
        # out = self.conv3_4(out)
        # out = _activate(out, self.bn3_4)
        conv3 = self.maxpool(conv3_3_act)

        # conv4_1 = self.conv4_1(conv3)
        # conv4_1_act = _activate(conv4_1, self.bn4_1)
        # conv4_2 = self.conv4_2(conv4_1_act)
        # conv4_2_act = _activate(conv4_2, self.bn4_2)
        # out = self.conv4_3(out)
        # out = _activate(out, self.bn4_3)
        # out = self.activation(self.bn4(out))
        # conv4 = self.maxpool(conv4_2_act)

        conv_out = conv3.reshape(_batch_size, -1)
        fc1 = self.activation(self.dropout(self.fc1(conv_out)))
        # fc2 = self.activation(self.dropout(self.fc2(fc1)))
        return self.output(fc1)


class ConvNet3D_Old(torch.nn.Module):
    """
    An fMRI slice given by BOLD5000 has shape: (71, 89, 72)
    """
    def __init__(self, num_outputs):
        super(ConvNet3D_Old, self).__init__()
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

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, x):
        _batch_size = x.shape[0]
        ### Convolution Layers
        # Current Size (batch_size, 1, 71, 89, 72)
        output = self.conv1(x)
        output = self.relu(output)
        output = self.bn1(output)

        # Current Size (batch_size, 16, 15, 19, 15)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.bn2(output)

        # Current Size (batch_size, 32, 7, 9, 7)
        output = self.conv3(output)
        output = self.relu(output)
        output = self.bn3(output)

        output = self.dropout1(output)
        output = output.reshape(_batch_size, -1)

        ### Fully Connected Layers
        output = self.linear1(output)
        output = self.dropout1(output)
        output = self.relu(output)
        output = self.linear2(output)

        return output
