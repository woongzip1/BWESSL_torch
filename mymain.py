import torchaudio as ta
import pytorch_lightning as pl
import torchaudio.transforms as T
import torch
import numpy as np
import yaml
import mir_eval
import gc

import warnings
# from train import RTBWETrain
# from datamodule import *
from utils import *

from tqdm import tqdm
import wandb
from pesq import pesq
from pystoi import stoi
import random
from torch.utils.data import Subset
import soundfile as sf
from datetime import datetime
import sys
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from SEANet import SEANet
from NewSEANet import NewSEANet
from MelGAN import Discriminator_MelGAN
from MBSTFTD import MultiBandSTFTDiscriminator
# from ssdiscriminatorblock import MultiBandSTFTDiscriminator


DEVICE = f'cuda' if torch.cuda.is_available() else 'cpu'

## Dictionary to store all models and information
TPARAMS = {}

NOTES = 'wavlm128_codec'
START_DATE = NOTES +'_' + datetime.now().strftime("%Y%m%d-%H%M%S")

def wandb_log(loglist, epoch, note):
    for key, val in loglist.items():
        if isinstance(val, torch.Tensor):
            item = val.cpu().detach().numpy()
        else:
            item = val

        try:
            if isinstance(item, float):
                log = item
            elif item.ndim in [2, 3]:  # 이미지 데이터
                log = wandb.Image(item)
            elif item.ndim == 1:  # 오디오 데이터
                log = wandb.Audio(item, sample_rate=16000, caption=f"{note.capitalize()} {key.capitalize()} Epoch {epoch}")
            else:
                log = item
        except Exception as e:
            print(f"Failed to log {key}: {e}")
            log = item

        wandb.log({
            f"{note.capitalize()} {key.capitalize()}": log,
        }, step=epoch)

# train_path_nb = "/home/woongjib/Projects/SSLBWE/sr/recon_mpeg/MP3_sr16_train_16kbps/test/p351/"
# train_path_wb = "/home/woongjib/Projects/SSLBWE/sr/VCTK/16Khz/test/p351"
# val_path_nb = "/home/woongjib/Projects/SSLBWE/sr/recon_mpeg/MP3_sr16_train_16kbps/test/p351/"
# val_path_wb = "/home/woongjib/Projects/SSLBWE/sr/VCTK/16Khz/test/p351"

train_path_nb = "/ssd2/woongzip/Datasets/recon_mpeg/MP3_sr16_train_16kbps/train"
train_path_wb = "/home/woongzip/VCTK/16Khz/train/"
val_path_nb = "/ssd2/woongzip/Datasets/recon_mpeg/MP3_sr16_train_16kbps/test"
val_path_wb = "/home/woongzip/VCTK/16Khz/test/"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility on cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("SEED!:",seed)

class CustomDataset(Dataset):
    def __init__(self, path_dir_nb, path_dir_wb, seg_len, iscodec=True, mode="train"):
        self.path_dir_nb = path_dir_nb
        self.path_dir_wb = path_dir_wb
        self.seg_len = seg_len
        self.mode = mode
        self.iscodec = iscodec

        set_seed(seed=42)

        paths_wav_wb = get_audio_paths(self.path_dir_wb, file_extensions='.flac')
        if self.iscodec:
            paths_wav_nb = get_audio_paths(self.path_dir_nb, file_extensions='.wav')
        else:
            paths_wav_nb = get_audio_paths(self.path_dir_nb, file_extensions='.flac')

        print(f"LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers loaded!")

        if len(paths_wav_wb) != len(paths_wav_nb):
            sys.exit(f"Error: LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers are different!")

        self.filenames = [(path_wav_wb, path_wav_nb) for path_wav_wb, path_wav_nb in zip(paths_wav_wb, paths_wav_nb)]
        print(f"{mode}: {len(self.filenames)} files loaded")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path_wav_wb, path_wav_nb = self.filenames[idx]

        wav_nb, sr_nb = ta.load(path_wav_nb)
        wav_wb, sr_wb = ta.load(path_wav_wb)

        wav_wb = wav_wb.view(1, -1)
        wav_nb = wav_nb.view(1, -1)

        if self.seg_len > 0 and self.mode == "train":
            duration = int(self.seg_len * 16000)
            sig_len = wav_wb.shape[-1]

            seed = idx
            np.random.seed(seed)

            t_start = np.random.randint(low=0, high=np.max([1, sig_len - duration - 2]), size=1)[0]
            if t_start % 2 == 1:
                t_start -= 1
            t_end = t_start + duration

            if self.iscodec:
                wav_nb = wav_nb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]
            else:
                wav_nb = wav_nb.repeat(1, t_end // sig_len + 1)[..., t_start // 2:t_end // 2]
            wav_wb = wav_wb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]

            wav_nb = self.ensure_length(wav_nb, sr_nb * self.seg_len)
            wav_wb = self.ensure_length(wav_wb, sr_wb * self.seg_len)

        elif self.mode == "val":
            min_len = min(wav_wb.shape[-1], wav_nb.shape[-1])
            wav_nb = self.ensure_length(wav_nb, min_len)
            wav_wb = self.ensure_length(wav_wb, min_len)

        else:
            sys.exit(f"unsupported mode! (train/val)")

        return wav_nb, wav_wb, get_filename(path_wav_wb)[0]

    @staticmethod
    def ensure_length(wav, target_length):
        if wav.shape[1] < target_length:
            pad_size = target_length - wav.shape[1]
            wav = F.pad(wav, (0, pad_size))
        elif wav.shape[1] > target_length:
            wav = wav[:, :target_length]
        return wav
#########################

def train_step(train_parameters):
    train_parameters['generator'].train()
    train_parameters['discriminator'].train()
    result = {}
    result['loss_G'] = 0
    result['loss_D'] = 0
    result['FM_loss'] = 0  # FM loss 결과 추가
    result['Mel_loss'] = 0  # Mel loss 결과 추가

    train_bar = tqdm(train_parameters['train_dataloader'], desc="Train", position=1, leave=False, disable=False)
    i = 0

    # Train DataLoader Loop
    # Epoch 1개만큼 train
    for lr, hr, _ in train_bar:
        i += 1
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)
        bwe = train_parameters['generator'](lr, hr)
        
        '''
        Train Generator
        '''        
        train_parameters['optim_G'].zero_grad()
        _, loss_GAN, loss_FM, loss_mel = train_parameters['discriminator'].loss_G(bwe, hr)
        ## MS Mel Loss
        loss_G = loss_GAN + 100*loss_FM + loss_mel
        loss_G.mean().backward()
        train_parameters['optim_G'].step()

        '''
        Discriminator
        '''
        train_parameters['optim_D'].zero_grad()
        loss_D = train_parameters['discriminator'].loss_D(bwe, hr)
        loss_D.mean().backward()
        train_parameters['optim_D'].step()
        
        result['loss_G'] += loss_G
        result['loss_D'] += loss_D
        result['FM_loss'] += loss_FM  # FM loss 추가
        result['Mel_loss'] += loss_mel  # Mel loss 추가

        train_bar.set_postfix({
                'Loss G': f'{loss_G.item():.2f}',
                'FM loss': f'{loss_FM.item():.2f}',
                'Mel loss': f'{loss_mel.item():.2f}',
                'Loss D': f'{loss_D.item():.2f}'
            })
        
        del lr, hr, bwe, loss_GAN, loss_FM, loss_mel, loss_D
        gc.collect()

    train_bar.close()
    result['loss_G'] /= len(train_parameters['train_dataloader'])
    result['loss_D'] /= len(train_parameters['train_dataloader'])
    result['FM_loss'] /= len(train_parameters['train_dataloader'])  # FM loss 평균 계산
    result['Mel_loss'] /= len(train_parameters['train_dataloader'])  # Mel loss 평균 계산

    

    return result

def test_step(test_parameters):
    test_parameters['generator'].eval()
    test_parameters['discriminator'].eval()
    result = {}
    result['loss_G'] = 0
    result['loss_D'] = 0
    result['FM_loss'] = 0  # FM loss 결과 추가
    result['Mel_loss'] = 0  # Mel loss 결과 추가
    result['PESQ'] = 0  # PESQ 결과 추가
    result['LSD'] = 0

    test_bar = tqdm(test_parameters['val_dataloader'], desc='Validation', position=1, leave=False, disable=False)
    i = 0
    total_pesq = 0
    total_lsd = 0
    # Test DataLoader Loop
    with torch.no_grad():
        for lr, hr, _ in test_bar:
            i += 1
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            bwe = test_parameters['generator'](lr, hr)
            
            _, loss_GAN, loss_FM, loss_mel = test_parameters['discriminator'].loss_G(bwe, hr)
            loss_G = loss_GAN + 100*loss_FM + loss_mel
            loss_D = test_parameters['discriminator'].loss_D(bwe, hr)
            
            result['loss_G'] += loss_G
            result['loss_D'] += loss_D
            result['FM_loss'] += loss_FM  # FM loss 추가
            result['Mel_loss'] += loss_mel  # Mel loss 추가

            pesq_score =  pesq(fs=16000, ref=hr.squeeze().cpu().numpy(), deg=bwe.squeeze().cpu().numpy(), mode="wb")
            total_pesq += pesq_score
            
            batch_lsd = lsd_batch(x_batch=hr.cpu(), y_batch=bwe.cpu())
            total_lsd += batch_lsd

            test_bar.set_postfix({
                'Loss G': f'{loss_G.item():.2f}',
                'FM loss': f'{loss_FM.item():.2f}',
                'Mel loss': f'{loss_mel.item():.2f}',
                'Loss D': f'{loss_D.item():.2f}',
                'PESQ': f'{pesq_score:.2f}',
                'LSD': f'{batch_lsd:.2f}' 
            })

             

            if i == 1:  # 맨 첫번째 배치에 대해서만 오디오 로그
                result['audio_lr'] = lr.squeeze().cpu().numpy()
                result['audio_hr'] = hr.squeeze().cpu().numpy()
                result['audio_bwe'] = bwe.squeeze().cpu().numpy()

            del lr, hr, bwe, loss_GAN, loss_FM, loss_mel, loss_D
            gc.collect()


        test_bar.close()
        result['loss_G'] /= len(test_parameters['val_dataloader'])
        result['loss_D'] /= len(test_parameters['val_dataloader'])
        result['FM_loss'] /= len(test_parameters['val_dataloader'])  # FM loss 평균 계산
        result['Mel_loss'] /= len(test_parameters['val_dataloader'])  # Mel loss 평균 계산
        result['PESQ'] = total_pesq / len(test_parameters['val_dataloader'])
        result['LSD'] = total_lsd / len(test_parameters['val_dataloader'])
        
    return result

def main():
    ################ Read Config Files
    config = yaml.load(open("./config_wavlm_128_codec.yaml", 'r'), Loader=yaml.FullLoader)
    # config = yaml.load(open("./config_baseline_codec.yaml", 'r'), Loader=yaml.FullLoader)

    wandb.init(project='SSLBWE_revised',
           entity='woongzip1',
           config=config,
           name=START_DATE,
           # mode='disabled',
           notes=NOTES)

    ################ Load Dataset
    train_dataset = CustomDataset(path_dir_nb=train_path_nb,path_dir_wb=train_path_wb,seg_len=2,mode="train")
    valid_dataset = CustomDataset(path_dir_nb=val_path_nb,path_dir_wb=val_path_wb,seg_len=2,mode="val")
    print(f'Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(valid_dataset)}\n')
    
    
    ################ Load DataLoader
    TPARAMS['train_dataloader'] = DataLoader(train_dataset, batch_size = config['dataset']['batch_size'], shuffle=True, 
                                             num_workers=config['dataset']['num_workers'], prefetch_factor=2, persistent_workers=True)
    TPARAMS['val_dataloader'] = DataLoader(valid_dataset, batch_size = 1, shuffle=False, 
                                             num_workers=config['dataset']['num_workers'], prefetch_factor=2, persistent_workers=True)  
     
    # print(f'DataLodaer Loaded!: {len(TPARAMS['train_dataloader'])} | {len(TPARAMS['val_dataloader'])}')
    print(f"DataLoader Loaded!: {len(TPARAMS['train_dataloader'])} | {len(TPARAMS['val_dataloader'])}")

    ################ Load Models
    warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated")
    warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")
    warnings.filterwarnings("ignore", message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")

    gen_type = config['model']['generator']

    print("########################################")
    if gen_type == "SEANet":
        TPARAMS['generator'] = SEANet(min_dim=8, causality=True)
        print(f"SEANet Generator: \n"
    f"                  Disc: {config['model']['discriminator']}")
    elif gen_type == "NewSEANet":
        TPARAMS['generator'] = NewSEANet(min_dim=8, 
                                       kmeans_model_path=config['model']['kmeans_path'],
                                       modelname=config['model']['sslname'],
                                       causality=True)
        print(f"NewSEANeT + {config['model']['sslname']} Generator: \n"
    f"                  kmeans:{os.path.splitext(os.path.basename(config['model']['kmeans_path']))[0]} \n"
    f"                  Disc: {config['model']['discriminator']},\n"
    f"                  SSL model: {config['model']['sslname']}")
    else: 
        raise ValueError(f"Unsupported generator type: {gen_type}")

    disc_type = config['model']['discriminator']
    if disc_type == "MSD":
        TPARAMS['discriminator'] = Discriminator_MelGAN()
    elif disc_type == "MBSTFTD":
        discriminator_config = config['model']['MultiBandSTFTDiscriminator_config']
        TPARAMS['discriminator'] = MultiBandSTFTDiscriminator(
            C=discriminator_config['C'],
            n_fft_list=discriminator_config['n_fft_list'],
            hop_len_list=discriminator_config['hop_len_list'],
            bands=discriminator_config['band_split_ratio'],
            ms_mel_loss_config=config['model']['ms_mel_loss_config']
        )
    else:
        raise ValueError(f"Unsupported discriminator type: {disc_type}")
    print("########################################")
    ################ Load Optimizers
    TPARAMS['optim_G'] = torch.optim.Adam(TPARAMS['generator'].parameters(), lr=config['optim']['learning_rate'], 
                                          betas=(config['optim']['B1'],config['optim']['B2']))
    TPARAMS['optim_D'] = torch.optim.Adam(TPARAMS['discriminator'].parameters(), config['optim']['learning_rate'], 
                                          betas=(config['optim']['B1'],config['optim']['B2']))
    
    ################ Load Checkpoint if available
    start_epoch = 1
    best_pesq = -1

    if config['train']['ckpt']:
        checkpoint_path = config['train']['ckpt_path']
        if os.path.isfile(checkpoint_path):
            start_epoch, best_pesq = load_checkpoint(TPARAMS['generator'], TPARAMS['discriminator'],
                                                     TPARAMS['optim_G'], TPARAMS['optim_D'], checkpoint_path)
        else:
            print(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")

    ################ Training Code
    print('Train Start!!')
    BAR = tqdm(range(start_epoch, config['train']['max_epochs'] + 1), position=0, leave=True)
    TPARAMS['generator'].to(DEVICE)
    TPARAMS['discriminator'].to(DEVICE)
    best_pesq = -1

    for epoch in BAR:
        TPARAMS['current_epoch'] = epoch
        train_result = train_step(TPARAMS)
        wandb_log(train_result, epoch, 'train')

        if epoch % config['train']['val_epoch'] == 0:
            # Validation step
            val_result = test_step(TPARAMS)
            wandb_log(val_result, epoch, 'val')

            if val_result['PESQ'] > best_pesq:
                best_pesq = val_result['PESQ']
                save_checkpoint(TPARAMS['generator'], TPARAMS['discriminator'], epoch, best_pesq, val_result['LSD'])

            desc = (f"Epoch [{epoch}/{config['train']['max_epochs']}] "
                    f"Loss G: {train_result['loss_G']:.2f}, "
                    f"FM Loss: {train_result['FM_loss']:.2f}, "
                    f"Mel Loss: {train_result['Mel_loss']:.2f}, "
                    f"Loss D: {train_result['loss_D']:.2f}, "
                    f"Val Loss G: {val_result['loss_G']:.2f}, "
                    f"Val FM Loss: {val_result['FM_loss']:.2f}, "
                    f"Val Mel Loss: {val_result['Mel_loss']:.2f}, "
                    f"Val Loss D: {val_result['loss_D']:.2f}, "
                    f"LSD: {val_result['LSD']:.2f}"
                    )
            
        else:
            desc = (f"Epoch [{epoch}/{config['train']['max_epochs']}] "
                    f"Loss G: {train_result['loss_G']:.2f}, "
                    f"FM Loss: {train_result['FM_loss']:.2f}, "
                    f"Mel Loss: {train_result['Mel_loss']:.2f}, "
                    f"Loss D: {train_result['loss_D']:.2f}"
                    )
        BAR.set_description(desc)

    gc.collect()
    final_epoch = config['train']['max_epochs']
    save_checkpoint(TPARAMS['generator'], TPARAMS['discriminator'], final_epoch, val_result['PESQ'], val_result['LSD'])

def save_checkpoint(generator, discriminator, epoch, pesq_score, lsd):
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_pesq_{pesq_score:.2f}.pth")
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': TPARAMS['optim_G'].state_dict(),  # Save optimizer state
        'optimizer_D_state_dict': TPARAMS['optim_D'].state_dict(),  # Save optimizer state
        'pesq_score': pesq_score,
        'lsd': lsd,
    }, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_pesq = checkpoint['pesq_score']
    
    if 'optimizer_G_state_dict' in checkpoint and 'optimizer_D_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    return start_epoch, best_pesq




if __name__ == "__main__":
    main()