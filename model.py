import os

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import optim


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, conditional_size=None, color_conditioning=False, **kwargs):
        assert mask_type in ['A', 'B']
        super(MaskConv2d, self).__init__(*args, **kwargs)
        self.conditional_size = conditional_size
        self.color_conditioning = color_conditioning
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                self.cond_op = nn.Linear(conditional_size[0], self.out_channels)
            else:
                self.cond_op = nn.Conv2d(conditional_size[0], self.out_channels, kernel_size=3, padding=1)

    def forward(self, x, cond=None):
        b = x.shape[0]
        out = nn.Conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                out = out + self.cond_op(cond).view(b, -1, 1, 1)
            else:
                out = out + self.cond_op(cond)
        return out

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if self.color_conditioning:
            assert self.in_channels % 3 == 0 and self.out_channels % 3 == 0
            one_third_in, one_third_out = self.in_channels // 3, self.out_channels // 3
            if mask_type == 'B':
                self.mask[:one_third_out, :one_third_in, k // 2, k // 2] = 1
                self.mask[one_third_out:2 * one_third_out, :2 * one_third_in, k // 2, k // 2] = 1
                self.mask[2 * one_third_out:, :, k // 2, k // 2] = 1
            else:
                self.mask[one_third_out:one_third_out * 2, :one_third_in, k // 2, k // 2] = 1
                self.mask[2 * one_third_out:, :2 * one_third_in, k // 2, k // 2] = 1
        else:
            if mask_type == 'B':
                self.mask[:, :, k // 2, k // 2] = 1


class ResBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.block = nn.ModuleList([
            nn.ReLU(),
            MaskConv2d('B', in_channels, in_channels // 2, 1, **kwargs),
            nn.ReLU(),
            MaskConv2d('B', in_channels // 2, in_channels // 2, 7, padding=3, **kwargs),
            nn.ReLU(),
            MaskConv2d('B', in_channels // 2, in_channels, 1, **kwargs)
        ])

    def forward(self, x, cond=None):
        out = x
        for layer in self.block:
            if isinstance(layer, MaskConv2d):
                out = layer(out, cond=cond)
            else:
                out = layer(out)
        return out + x


class LayerNorm(nn.LayerNorm):
    def __init__(self, color_conditioning, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_conditioning = color_conditioning

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.color_conditioning:
            x = x.contiguous().view(*(x_shape[:-1] + (3, -1)))
        x = super().forward(x)
        if self.color_conditioning:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()


class PixelCNN(nn.Module):
    def __init__(self, input_shape, n_colors, n_filters=64, kernel_size=7, n_layer=5, conditional_size=None,
                 use_resblock=False, color_conditioning=False):
        super().__init__()
        assert n_layer >= 2
        n_channels = input_shape[0]
        kwargs = dict(conditional_size=conditional_size, color_conditioning=color_conditioning)
        if use_resblock:
            block_init = lambda: ResBlock(n_filters, **kwargs)
        else:
            block_init = lambda: MaskConv2d('B', n_filters, n_filters, kernel_size=kernel_size,
                                            padding=kernel_size // 2, **kwargs)
        model = nn.ModuleList(
            [MaskConv2d('A', n_channels, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, **kwargs)])
        for _ in range(n_layer):
            if color_conditioning:
                model.append(LayerNorm(color_conditioning, n_filters // 3))
            else:
                model.append(LayerNorm(color_conditioning, n_filters))
            model.append(nn.ReLU())
            model.append(block_init())
        model.append(nn.ReLU())
        model.append(MaskConv2d('B', n_filters, n_filters, 1, **kwargs))
        model.append(nn.ReLU())
        model.append(MaskConv2d('B', n_filters, n_colors * n_channels, 1, **kwargs))
        if conditional_size:
            if len(conditional_size) == 1:
                self.cond_op = lambda x: x  # no processing if one hot
            else:
                # fro grayscale pixelcnn
                self.cond_op = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 64, 3, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 64, 3, padding=1),
                                             nn.ReLU())
        self.net = model
        self.input_shape = input_shape
        self.n_colors = n_colors
        self.n_channels = n_channels
        self.color_conditioning = color_conditioning
        self.conditional_size = conditional_size

    def forward(self, x, cond=None):
        b = x.shape[0]
        out = (x.float() / (self.n_colors - 1) - 0.5) / 0.5
        if self.conditional_size:
            cond = self.cond_op(cond)
        for layer in self.net:
            if isinstance(layer, MaskConv2d) or isinstance(layer, ResBlock):
                out = layer(out, cond=cond)
            else:
                out = layer(out)
        if self.color_conditioning:
            return out.view(b, self.n_channels, self.n_colors, *self.input_shape[1:]).permute(0, 2, 1, 3, 4)
        else:
            return out.view(b, self.n_colors, *self.input_shape)

    def loss(self, x, cond=None):
        return F.cross_entropy(self.forward(x, cond), x.long())


def train(model, train_loader, optimizer, epoch, grad_clip=None):
    model.train()

    train_losses = []
    for x, _ in train_loader:
        # x = x.cuda().contiguous()
        loss = model.loss(x)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses


def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in data_loader:
            # x = x.cuda().contiguous()
            loss = model.loss(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss.item()


quiet = False


def train_epochs(model, train_loader, test_loader, train_args):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = [eval_loss(model, test_loader)]
    for epoch in range(epochs):
        model.train()
        train_losses.extend(train(model, train_loader, optimizer, epoch, grad_clip))
        test_loss = eval_loss(model, test_loader)
        test_losses.append(test_loss)
        if not quiet:
            print(f'Epoch {epoch}, Test loss {test_loss:.4f}')

    return train_losses, test_losses


if __name__ == '__main__':
    mnist_data = torchvision.datasets.MNIST(os.path.dirname(__file__), download=False,
                                            transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(mnist_data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=4)
    model = PixelCNN((1, 28, 28), 2, n_layer=5)
    for images, _ in train_loader:
        x = model(images)
        print(0)
        break
