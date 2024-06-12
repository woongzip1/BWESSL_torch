# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm
# import torchaudio
# from loss import ms_mel_loss

# from einops import rearrange
# import typing as tp
# from loss import ms_mel_loss

# def WNConv2d(*args, **kwargs):
#     """
#     2d-convolution with wieght normalization
#     """
#     return weight_norm(nn.Conv2d(*args, **kwargs))

# class MultiBandSTFTDiscriminator(nn.Module):
#     def __init__(self, C: int, n_fft_list: list, hop_len_list: list, bands: list, name: str = "", **kwargs):
#         super().__init__()
#         self.name = name
#         self.bands = bands
#         self.n_fft_list = n_fft_list
#         self.hop_len_list = hop_len_list

#         ch = C
#         self.layers_per_band = nn.ModuleList()
#         for n_fft, hop_length in zip(n_fft_list, hop_len_list):
#             band_layers = nn.ModuleList([
#                 WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
#                 WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
#                 WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
#                 WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
#                 WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
#             ])
#             self.layers_per_band.append(band_layers)
        
#         self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1))
#         self.activation = nn.LeakyReLU(0.1)
    
#     def stft_band_split(self, x):
#         # print(f"Band Split! X size: {x.size()}")
#         if len(x.size()) == 1:  # Check if the input is 1D (single sample)
#             x = x.unsqueeze(0)  # Add batch and channel dimensions
        

#         x_bands = []
#         # print(f"{x.size()} X size changed")
#         for n_fft, hop_length in zip(self.n_fft_list, self.hop_len_list):
#             window = torch.hann_window(n_fft).to(x.device)  # Create a Hann window
#             x_stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, 
#                                 window=window, return_complex=True)
#             x_stft = torch.view_as_real(x_stft)
#             # print(f"{x_stft.size()}x_stft size! ")
#             x_stft = rearrange(x_stft, "b f t c -> b c t f")
#             band_splits = [x_stft[..., int(b[0] * (n_fft // 2 + 1)):int(b[1] * (n_fft // 2 + 1))] for b in self.bands]
#             x_bands.extend(band_splits)
#         return x_bands

#     def forward(self, x, return_features=False):
#         x_bands = self.stft_band_split(x)
#         feature_map = []
#         outputs_per_band = []

#         for band, layers in zip(x_bands, self.layers_per_band):
#             for layer in layers:
#                 band = layer(band)
#                 band = self.activation(band)
#                 if return_features:
#                     feature_map.append(band)
#             outputs_per_band.append(band)

#         z = torch.cat(outputs_per_band, dim=-1)
#         z = self.conv_post(z)
        
#         return z, feature_map

#     def loss_D(self, x_proc, x_orig, *args, **kwargs):
#         x_proc = x_proc.squeeze()[..., :x_orig.shape[-1]].detach()
#         x_orig = x_orig.squeeze()[..., :x_proc.shape[-1]]

#         D_proc, _ = self(x_proc)
#         D_orig, _ = self(x_orig)

#         loss = torch.relu(1 - D_orig).mean() + torch.relu(1 + D_proc).mean()
#         return loss

#     def loss_G(self, x_proc, x_orig, *args, **kwargs):
#         x_proc = x_proc.squeeze()[..., :x_orig.shape[-1]]
#         x_orig = x_orig.squeeze()[..., :x_proc.shape[-1]]

#         D_proc, F_proc = self(x_proc, return_features=True)
#         D_orig, F_orig = self(x_orig, return_features=True)

#         loss_GAN = torch.relu(1 - D_proc).mean()

#         loss_FM = sum(torch.mean(torch.abs(f_p - f_o.detach())) for f_p, f_o in zip(F_proc, F_orig)) / len(F_proc)
#         loss_FM = 100 * loss_FM

#         # loss_ms_mel = ms_mel_loss(x_orig, x_proc, **self.config['model']['ms_mel_loss_config'])

#         loss = loss_GAN + loss_FM 
#         ## Output three losses
#         return loss, loss_GAN, loss_FM

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torchaudio
from loss import ms_mel_loss

from einops import rearrange
import typing as tp
from loss import ms_mel_loss

def WNConv2d(*args, **kwargs):
    """
    2d-convolution with wieght normalization
    """
    return weight_norm(nn.Conv2d(*args, **kwargs))

class MultiBandSTFTDiscriminator(nn.Module):
    def __init__(self, C: int, n_fft_list: list, hop_len_list: list, bands: list, ms_mel_loss_config: dict, name: str = "", **kwargs):
        super().__init__()
        self.name = name
        self.bands = bands
        self.n_fft_list = n_fft_list
        self.hop_len_list = hop_len_list
        self.ms_mel_loss_config = ms_mel_loss_config

        ch = C
        self.layers_per_band = nn.ModuleList()
        for n_fft, hop_length in zip(n_fft_list, hop_len_list):
            band_layers = nn.ModuleList([
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ])
            self.layers_per_band.append(band_layers)
        
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1))
        self.activation = nn.LeakyReLU(0.1)
    
    def stft_band_split(self, x):
        # print(f"Band Split! X size: {x.size()}")
        if len(x.size()) == 1:  # Check if the input is 1D (single sample)
            x = x.unsqueeze(0)  # Add batch and channel dimensions
        

        x_bands = []
        # print(f"{x.size()} X size changed")
        for n_fft, hop_length in zip(self.n_fft_list, self.hop_len_list):
            window = torch.hann_window(n_fft).to(x.device)  # Create a Hann window
            x_stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, 
                                window=window, return_complex=True)
            x_stft = torch.view_as_real(x_stft)
            # print(f"{x_stft.size()}x_stft size! ")
            x_stft = rearrange(x_stft, "b f t c -> b c t f")
            band_splits = [x_stft[..., int(b[0] * (n_fft // 2 + 1)):int(b[1] * (n_fft // 2 + 1))] for b in self.bands]
            x_bands.extend(band_splits)
        return x_bands

    def forward(self, x, return_features=False):
        x_bands = self.stft_band_split(x)
        feature_map = []
        outputs_per_band = []

        for band, layers in zip(x_bands, self.layers_per_band):
            for layer in layers:
                band = layer(band)
                band = self.activation(band)
                if return_features:
                    feature_map.append(band)
            outputs_per_band.append(band)

        z = torch.cat(outputs_per_band, dim=-1)
        z = self.conv_post(z)
        
        return z, feature_map

    def loss_D(self, x_proc, x_orig, *args, **kwargs):
        x_proc = x_proc.squeeze()[..., :x_orig.shape[-1]].detach()
        x_orig = x_orig.squeeze()[..., :x_proc.shape[-1]]

        D_proc, _ = self(x_proc)
        D_orig, _ = self(x_orig)

        loss = torch.relu(1 - D_orig).mean() + torch.relu(1 + D_proc).mean()
        return loss

    def loss_G(self, x_proc, x_orig, *args, **kwargs):
        x_proc = x_proc.squeeze()[..., :x_orig.shape[-1]]
        x_orig = x_orig.squeeze()[..., :x_proc.shape[-1]]

        D_proc, F_proc = self(x_proc, return_features=True)
        D_orig, F_orig = self(x_orig, return_features=True)

        loss_GAN = torch.relu(1 - D_proc).mean()
        loss_FM = sum(torch.mean(torch.abs(f_p - f_o.detach())) for f_p, f_o in zip(F_proc, F_orig)) / len(F_proc)
        loss_ms_mel = ms_mel_loss(x_orig, x_proc, **self.ms_mel_loss_config)

        loss = loss_GAN + 100*loss_FM + loss_ms_mel 
        ## Output three losses
        return loss, loss_GAN, loss_FM, loss_ms_mel


