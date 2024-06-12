import os
import shutil
import argparse
import torch
import numpy as np
from scipy.signal import stft
from matplotlib import pyplot as plt
import librosa

""" 주어진 디렉토리에서 지정된 확장자를 가진 모든 오디오 파일의 절대 경로를 반환합니다. """
def get_audio_paths(paths: list, file_extensions=['.wav', '.flac']):
    audio_paths = []
    if isinstance(paths, str):
        paths = [paths]
        
    for path in paths:
        for root, dirs, files in os.walk(path):
            audio_paths += [os.path.join(root, file) for file in files if os.path.splitext(file)[-1].lower() in file_extensions]
                        
    audio_paths.sort(key=lambda x: os.path.split(x)[-1])
    
    return audio_paths

def count_audio_files(paths: list, file_extensions=['.wav', '.mp3', '.flac']):
    """ 주어진 디렉토리에서 지정된 확장자를 가진 모든 오디오 파일의 개수를 반환합니다. """
    audio_files = get_audio_paths(paths, file_extensions)
    return len(audio_files)

def check_dir_exist(path_list):
    if type(path_list) == str:
        path_list = [path_list]
        
    for path in path_list:
        if type(path) == str and os.path.splitext(path)[-1] == '' and not os.path.exists(path):
            os.makedirs(path)       

def get_filename(path):
    return os.path.splitext(os.path.basename(path))  

"""input LR path -> LR/train & LR/test""" 
def path_into_traintest(lr_folder_path):
    # LR 폴더 내의 모든 하위 폴더(화자 폴더)를 리스트로 가져옵니다.
    speaker_folders = sorted([f for f in os.listdir(lr_folder_path) if os.path.isdir(os.path.join(lr_folder_path, f))])
    
    # 마지막 9명의 화자를 테스트 세트로 설정합니다.
    test_speakers = speaker_folders[-9:]
    train_speakers = speaker_folders[:-9]

    # Train과 Test 폴더를 생성합니다.
    train_path = os.path.join(lr_folder_path, 'train')
    test_path = os.path.join(lr_folder_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # 트레이닝 폴더에 화자 폴더를 이동합니다.
    for speaker in train_speakers:
        original_path = os.path.join(lr_folder_path, speaker)
        destination_path = os.path.join(train_path, speaker)
        shutil.move(original_path, destination_path)

    # 테스트 폴더에 화자 폴더를 이동합니다.
    for speaker in test_speakers:
        original_path = os.path.join(lr_folder_path, speaker)
        destination_path = os.path.join(test_path, speaker)
        shutil.move(original_path, destination_path)

    print(f"Train set and test set have been created at {train_path} and {test_path}.")

def si_sdr(reference_signal, estimated_signal):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Parameters:
    - reference_signal (torch.Tensor): The reference signal (N, L)
    - estimated_signal (torch.Tensor): The estimated signal (N, L)
    
    Returns:
    - si_sdr (torch.Tensor): The SI-SDR value for each signal in the batch (N,)
    """
    # Ensure the inputs are of the same shape
    assert reference_signal.shape == estimated_signal.shape, "Input and reference signals must have the same shape"
    
    # Check if Shape is -> N x L
    if reference_signal.dim() == 3:
        reference_signal = reference_signal.squeeze(1)
        estimated_signal = estimated_signal.squeeze(1)
    
    # Compute the scaling factor
    reference_signal = reference_signal - reference_signal.mean(dim=1, keepdim=True)
    estimated_signal = estimated_signal - estimated_signal.mean(dim=1, keepdim=True)
    
    s_target = (torch.sum(reference_signal * estimated_signal, dim=1, keepdim=True) / torch.sum(reference_signal ** 2, dim=1, keepdim=True)) * reference_signal
    
    e_noise = estimated_signal - s_target
    
    si_sdr_value = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / torch.sum(e_noise ** 2, dim=1))
    
    return si_sdr_value


def lsd_batch(x_batch, y_batch, fs=16000, frame_size=0.02, frame_shift=0.02):
    # 프레임 크기와 프레임 이동 크기를 샘플 단위로 변환
    frame_length = int(frame_size * fs)
    frame_step = int(frame_shift * fs)
   
    # 배치 크기와 입력 길이
    # print(x_batch.shape)
    # print(y_batch.shape)
    if isinstance(x_batch, np.ndarray):
        x_batch = torch.from_numpy(x_batch)
        y_batch = torch.from_numpy(y_batch)
   
    if x_batch.dim()==1:
        batch_size = 1
    ## Batch가 1이라 1 x 32000이 나온다면
 
    if x_batch.dim()==2:
        x_batch=x_batch.unsqueeze(1)
    batch_size, _, signal_length = x_batch.shape
   
    if y_batch.dim()==1:
        y_batch=y_batch.reshape(batch_size,1,-1)
    if y_batch.dim()==2:
        y_batch=y_batch.unsqueeze(1)
   
    lsd_values = []
 
    # print(type(x_batch))
    # print(x_batch.size())
    # print(y_batch.size())
    # print(x_batch.dim())
    for i in range(batch_size):
        x = x_batch[i, 0, :].numpy()
        # print(type(y_batch))
       
        # print(y_batch.size())
        y = y_batch[i, 0, :].numpy()
 
        # print(x.shape, y.shape,"hihi")
        # STFT 계산
        f_x, t_x, Zxx_x = stft(x, fs, nperseg=frame_length, noverlap=frame_length - frame_step)
        f_y, t_y, Zxx_y = stft(y, fs, nperseg=frame_length, noverlap=frame_length - frame_step)
       
        # 파워 스펙트럼 계산
        power_spec_x = np.abs(Zxx_x) ** 2
        power_spec_y = np.abs(Zxx_y) ** 2
       
        # 로그 스펙트럼 계산
        log_spec_x = np.log(power_spec_x + 1e-10)  # 작은 값을 더해 로그 계산 시 0을 피함
        log_spec_y = np.log(power_spec_y + 1e-10)
       
        # 로그 스펙트럼 차이의 제곱 계산
        lsd = np.sqrt(np.mean((log_spec_x - log_spec_y) ** 2, axis=0))
       
        # 프레임들 간에 평균
        mean_lsd = np.mean(lsd)
       
        lsd_values.append(mean_lsd)
   
    # 배치의 평균 LSD 값 계산
    batch_mean_lsd = np.mean(lsd_values)
   
    return batch_mean_lsd

def draw_spec(x,
              figsize=(10, 6), title='', n_fft=2048,
              win_len=1024, hop_len=256, sr=16000, cmap='inferno',
              vmin=-50, vmax=40, use_colorbar=True,
              ylim=None,
              title_fontsize=10,
              label_fontsize=8,
                return_fig=False,
                save_fig=False, save_path=None):
    fig = plt.figure(figsize=figsize)
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    stft = 20 * np.log10(np.clip(np.abs(stft), a_min=1e-8, a_max=None))

    plt.imshow(stft,
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', extent=[0, len(x) / sr, 0, sr//2])

    if use_colorbar:
        plt.colorbar()

    plt.xlabel('Time (s)', fontsize=label_fontsize)
    plt.ylabel('Frequency (Hz)', fontsize=label_fontsize)

    if ylim is None:
        ylim = (0, sr / 2)
    plt.ylim(*ylim)

    plt.title(title, fontsize=title_fontsize)
    
    if save_fig:
        plt.savefig(f"{save_path}.png")
    
    if return_fig:
        plt.close()
        return fig
    else:
        return stft
        plt.show()
        
        





def main():
    parser = argparse.ArgumentParser(description="Path for train test split")
    parser.add_argument("--path", type=str, help="Path for dataset")
    args = parser.parse_args()
    path_into_traintest(args.path)

if __name__ == "__main__":
    main()

