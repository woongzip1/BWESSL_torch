import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.signal.windows import hann

# from soundstream.balancer import *


def loss_aggregator(x, x_hat, loss_name_list, loss_weight_dict, train_config, **kwargs):
    """
    Aggregate different reconstruction loss functions together. (loss from the discriminators are not included!)
    
    Args:
        x (Tensor): input of the model.
        x_hat (Tensor): output of the model, which is the reconstructed input.
        loss_name_list (list): list of the name of loss functions.
        loss_weight_dict (dict): Dictionary that has loss name as key and weight of the loss as value.
        train_config (dict): configuration dictionary for model training
    """
    loss_dict = {}
    loss_report = {}
    
    if 'L1' in loss_name_list:
        l1_config = train_config['criteria']['L1']
        l1_loss = L1_distance(x, x_hat, reduction=l1_config['reduction'])
        loss_dict['L1'] = l1_loss
        loss_report['L1'] = l1_loss.item()
    
    if 'L2' in loss_name_list:
        l2_config = train_config['criteria']['L2']
        l2_loss = L2_distance(x, x_hat, reduction=l2_config['reduction'])
        loss_dict['L2'] = l2_loss
        loss_report['L2'] = l2_loss.item()
    
    if 'SED' in loss_name_list:
        sed_config = train_config['criteria']['SED']
        sed_loss = ms_spectral_energy_distance(x, x_hat, sr=kwargs['sr'], eps=1e-16, return_all=False, **sed_config)
        loss_dict['SED'] = sed_loss
        loss_report['SED'] = sed_loss.item()    
    
    if 'MSMEL' in loss_name_list:
        msmel_config = train_config['criteria']['MSMEL']
        msmel_loss = ms_mel_loss(x, x_hat, sr=kwargs['sr'], **msmel_config)
        loss_dict['MSMEL'] = msmel_loss
        loss_report['MSMEL'] = msmel_loss.item() 
    
    return loss_dict, loss_report

def loss_backward(loss_dict, loss_name_list, loss_weight_dict, balancer, balancer_input=None, balancer_update_params=None, retain_graph=False, gradient_accum_every=1, **kwargs):    
    """
    Calculate weighted sum of the loss functions, and backpropagate the sum.
    Generator adversarial loss and feature matching loss are included.
    VQ commitment loss is backpropagated separately, since the commit loss is calculated on the latent space.
    If use loss balancer, than only the loss functions that contains reconstructed signal (output of the autoencoder) in the forward pass
    are rescalecd and backpropagated using the loss balancer.
    
    Args:
        loss_dict (dict): dictionary that has loss name as key  and calculated loss value, represented in torch.Tensor, as value.
        loss_name_list (list): list of the name of loss functions.
        loss_weight_dict (dict): dictionary that has loss name as key and weight of the loss as value.
        train_config (dict): configuration dictionary for model training
        balancer (class Balancer): loss balancer object if not None.
        balancer_input (torch.Tensor): input of the balancer. Every loss rescaled by the balancer must include the input tensor in the forward pass.
        balancer_update_params (torch.Tensor): if not None, than the only learnable parameters transmitted by 'balancer_update_params' recieve backpropagated gradients.
                                                if None, than calculate gradient for every parameters behind the 'balancer_input'.
        retain_graph (bool): if true, than a graph that tracks the gradient flow will remain after the backward().
        gradient_accum_every (int): the number of backpropagation needed for one optimizer step. If larger than 1, than need to rescale loss values.
    """
    # Backpropagate auxiliary losses first! (commitment loss, codebook loss, entropy loss, ...)
    loss_device = list(loss_dict.values())[0].device
    aux_loss_sum = torch.Tensor([0]).to(loss_device)
    loss_dict_keys = list(loss_dict.keys())
    for key in loss_dict_keys:
        if key in ['commit', 'codebook', 'entropy']:
            aux_loss = loss_dict.pop(key)
            aux_loss_sum = aux_loss_sum + aux_loss
            
    aux_loss_sum = aux_loss_sum / gradient_accum_every
    aux_loss_sum.requires_grad_(True)
    aux_loss_sum.backward(retain_graph=True)      
    
    # Calculate weighted sum of the losses (if not using Balancer)
    total_recon_loss = 0
    for loss_name, loss in loss_dict.items():
        total_recon_loss += loss * loss_weight_dict[loss_name] / gradient_accum_every
    
    # Backpropagation
    if balancer is None:
        total_recon_loss.backward(retain_graph=retain_graph)
    else:
        balancer.backward(losses=loss_dict, input=balancer_input, params=balancer_update_params, retain_graph=retain_graph)
    

def L1_distance(x, x_hat, reduction='mean'):
    """
    Calculate L1 distance between two signals.
    
    Args:
        x (Tensor) [B, ..., T]: ground truth waveform
        x_hat (Tensor) [B, ..., T]: generated waveform
        reduction (str): 'mean' (averaged over all dimensions), 'sum' (summed over dimensions except the batchwise dimension)
        
    Returns:
        loss (Tensor) [1]: L1 loss
    """
    loss = torch.sum(torch.abs(x - x_hat))
    
    if reduction == 'mean':
        loss /= torch.numel(x)
    elif reduction == 'sum':
        loss /= x.shape[0]
    
    return loss

def L2_distance(x, x_hat, reduction='mean', is_sqrt=True, eps=1e-16):
    """
    Args:
        x (Tensor) [B, ..., T]: ground truth waveform
        x_hat (Tensor) [B, ..., T]: generated waveform
        reduction (str): 'mean' (averaged over all dimensions), 'sum' (summed over dimensions except the batchwise dimension)
    Returns:
        loss (Tensor) [1]: L2 loss
    """
    loss = torch.sum((x - x_hat) ** 2, dim=-1)
    if is_sqrt:
        loss = torch.sqrt(loss + eps)
    loss = torch.sum(loss)
    
    if reduction == 'mean':
        loss /= torch.numel(x)
    elif reduction == 'sum':
        loss /= x.shape[0]
    
    return loss
        
        
def ms_spectral_energy_distance(x, x_hat, use_mel=True, sr=44100, n_fft_for_mel=2048, mel_bin=64, reduction='mean', exp_min=6,  exp_max=11, hop_ratio=0.25, power=1, normalize=False, alpha_type='adaptive', eps=1e-16, return_all=False, **kwargs):
    """
    Multi-scale spectral energy distance loss
    References:
        Alexey Gritsenko et al., "A Spectral Energy Distance for Parallel Speech Synthesis", NeurIPS, 2020.
        Neil Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec", IEEE/ACM Transactions on Audio, Speech, and Language Processing 30, 2021.
        
    Args:
        x (torch.Tensor) [B, ..., T]: ground truth waveform
        x_hat (torch.Tensor) [B, ..., T]: generated waveform
        use_mel (bool): True (Transform STFT into mel-spectrogram and calculate loss), False (calculate loss over linear STFT directly)
        sr (int): sampling rate for mel-filterbank calculation
        n_fft_for_mel (int): n_fft of STFT when calculating linear spectrogram before applying mel-filterbank
        mel_bin (int): the number of bins of the mel-spectrogram
        reduction (str): 'mean' (averaged over all dimensions), 'sum' (summed over dimensions except the batchwise dimension)
        exp_min (int): the shortest window length of STFT in a power of two     win_len = 2^(idx_min), 2^(idx_min+1), ..., 2^(idx_max)
        exp_max (int): the shortest window length of STFT in a power of two 
        hop_ratio(float): ratio between hop length and window length    hop_length = int(win_len * hop_ratio)
        power(int): 2 (calculate mel-spectrogram from the power spectrogram if use_mel is true, or calculate loss over power spectrogram if use_mel is false), 
                    1 (calculate mel-spectrogram from the magintude spectrogram if use_mel is true, or calculate loss over magnitude spectrogram if use_mel is false)
        normalize (bool): True (divede the linear spectrogram with win_len), False (no normalize for spectrogram calculation)
        alpha_type (str): 'adaptive' (alpha = sqrt(win_len / 2)), 'fixed' (alpha = 1)
        eps (float): small additive value for log calcuation
        
    Returns:
        loss (torch.Tensor) [1]: loss value
    """
    loss_total = 0
    loss_l1 = 0
    loss_l2 = 0

    normalize_type = 'frame_length' if normalize is True else False
    
    for win_len_idx in range(exp_min, exp_max + 1):
        win_len = int(2 ** win_len_idx)
        hop_len = int(win_len * hop_ratio)
        
        if alpha_type == 'adaptive':
            alpha = math.sqrt(win_len / 2)
        elif alpha_type == 'fixed':
            alpha = 1
        
        if use_mel:
            # Mel-spectrogram
            sig_to_spg = T.Spectrogram(n_fft=n_fft_for_mel, win_length=win_len, hop_length=hop_len, power=power, normalized=normalize_type).to(x.device)
            spg_to_mel = T.MelScale(n_mels=mel_bin, sample_rate=sr, n_stft=n_fft_for_mel//2+1).to(x.device)
            x_stft = spg_to_mel(sig_to_spg(x))  # [B, C, mels, T]
            x_hat_stft = spg_to_mel(sig_to_spg(x_hat))
        else:
            # Linear spectrogram
            sig_to_spg = T.Spectrogram(n_fft=win_len, win_length=win_len, hop_length=hop_len, power=power, normalized=normalize_type).to(x.device)
            x_stft = sig_to_spg(x)  # [B, C, freq_bins, T]
            x_hat_stft = sig_to_spg(x_hat)
        
        # Calculate loss
        l1_term = torch.sum(torch.abs(x_stft - x_hat_stft))
        l2_term = torch.log(x_stft + eps) - torch.log(x_hat_stft + eps)
        l2_term = (torch.sum((l2_term ** 2), dim=-2) + eps) ** 0.5
        l2_term = torch.sum(l2_term)
        
        if reduction == 'mean':
            l1_term /= torch.numel(x_stft)
            l2_term /= torch.numel(x_stft)
        elif reduction == 'sum':
            l1_term /= x_stft.shape[0]
            l2_term /= x_stft.shape[0]
            
        loss_l1 += l1_term
        loss_l2 += alpha * l2_term
    
    loss_total = loss_l1 + loss_l2
    
    if return_all:
        return loss_total, loss_l1, loss_l2
    else:
        return loss_total
    

def ms_mel_loss(x, x_hat, n_fft_list=[32, 64, 128, 256, 512, 1024, 2048], hop_ratio=0.25, 
                mel_bin_list=[5, 10, 20, 40, 80, 160, 320], fmin=0, fmax=None, 
                sr=44100, mel_power=1.0, eps=1e-5, reduction='sum', loss_ratio=1.0, **kwargs):
    """
    Multi-scale spectral energy distance loss
    References:
        Kumar, Rithesh, et al. "High-Fidelity Audio Compression with Improved RVQGAN." NeurIPS, 2023.
    Args:
        x (torch.Tensor) [B, ..., T]: ground truth waveform
        x_hat (torch.Tensor) [B, ..., T]: generated waveform
        n_fft_list (List of int): list that contains n_fft for each scale
        hop_ratio (float): hop_length = n_fft * hop_ratio
        mel_bin_list (List of int): list that contains the number of mel bins for each scale
        sr (int): sampling rate
        fmin (float): minimum frequency for mel-filterbank calculation
        fmax (float): maximum frequency for mel-filterbank calculation
        mel_power (float): power to raise magnitude to before taking log
    Returns:
    """
    
    assert len(n_fft_list) == len(mel_bin_list)

    loss = 0
    
    for n_fft, mel_bin in zip(n_fft_list, mel_bin_list):
        sig_to_spg = T.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=int(n_fft * hop_ratio), 
                                    window_fn=hann, wkwargs={"sym": False},\
                                    power=1.0, normalized=False, center=True).to(x.device)
        spg_to_mel = T.MelScale(n_mels=mel_bin, sample_rate=sr, n_stft=n_fft//2+1, f_min=fmin, 
                                f_max=fmax, norm="slaney", mel_scale="slaney").to(x.device)  
        x_mel = spg_to_mel(sig_to_spg(x))  # [B, C, mels, T]
        x_hat_mel = spg_to_mel(sig_to_spg(x_hat))
        
        log_term = torch.sum(torch.abs(x_mel.clamp(min=eps).pow(mel_power).log10() - x_hat_mel.clamp(min=eps).pow(mel_power).log10()))
        
        if reduction == 'mean':
            log_term /= torch.numel(x_mel)
        elif reduction == 'sum':
            log_term /= x_mel.shape[0]
        
        loss += log_term
        
    loss *= loss_ratio
    
    return loss

def ms_stft_loss(x, x_hat, n_fft_list=[512, 2048], hop_ratio=0.25, eps=1e-5, reduction='sum', **kwargs):
    """
    Multi-scale spectral energy distance loss
    References:
        Kumar, Rithesh, et al. "High-Fidelity Audio Compression with Improved RVQGAN." NeurIPS, 2023.
    Args:
        x (torch.Tensor) [B, ..., T]: ground truth waveform
        x_hat (torch.Tensor) [B, ..., T]: generated waveform
        n_fft_list (List of int): list that contains n_fft for each scale
        hop_ratio (float): hop_length = n_fft * hop_ratio
    Returns:
    """
    
    loss = 0
    
    for n_fft in n_fft_list:
        sig_to_spg = T.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=int(n_fft * hop_ratio), window_fn=hann, wkwargs={"sym": False},\
                                    power=1.0, normalized=False, center=True).to(x.device)
        x_spg = sig_to_spg(x)  # [B, C, F, T]
        x_hat_spg = sig_to_spg(x_hat)
        
        log_term = torch.sum(torch.abs(x_spg.clamp(min=eps).log10() - x_hat_spg.clamp(min=eps).log10()))
        
        if reduction == 'mean':
            log_term /= torch.numel(x_spg)
        elif reduction == 'sum':
            log_term /= x_spg.shape[0]
        
        loss += log_term
        
    return loss