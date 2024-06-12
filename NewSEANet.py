import torch as th
import torch 
import torch.nn as nn
import torch.nn.functional as F
from SEANet import EncBlock,DecBlock,Conv1d,ConvTransposed1d,Pad
import pickle
from transformers import HubertModel, AutoProcessor, Wav2Vec2Model, WavLMModel, AutoModel

import warnings


def HuBERT_layer(model, processor, audio, modelname='hubert'):
    # audio의 끝에 80개의 zero를 pad
    audio = F.pad(audio, (0, 80), "constant", 0)
        
    # 장치를 확인하고 입력 데이터를 올바른 장치로 전송
    dev = audio.device
    # print("Input device:", dev)
    # print("Model device:", next(model.parameters()).device)  # 모델 파라미터의 장치 확인

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(dev)
    
    # print("*************************************")
    # print("Input HuBERT Shape Before Squeeze : ", inputs.shape)
    # print("*************************************")
    ## input shape: 1 x B x L ---> B x L
    
    inputs = inputs.squeeze(0) 
    
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        
    if modelname=='hubert':
        out_layer = outputs.hidden_states[22].to(dev)
    elif modelname=='w2v':
        out_layer = outputs.hidden_states[24].to(dev)
    elif modelname=='wavlm':
        out_layer = outputs.hidden_states[22].to(dev)
    
    # print(out_layer.shape,"haha")
    return out_layer

class NewSEANet(nn.Module):
    # def __init__(self, min_dim=8,kmeans_model_path='/home/woongzip/RealTimeBWE/Kmeans/kmeans_modelweight_200.pkl', **kwargs):
    def __init__(self, min_dim=8,kmeans_model_path=None, modelname="hubert",**kwargs):

        super().__init__()
        
        ## Load Kmeans model
        with open(kmeans_model_path, 'rb') as file:
            self.kmeans = pickle.load(file)
                
        self.min_dim = min_dim
        
        self.conv_in = Conv1d(
            in_channels = 1,
            out_channels = min_dim,
            kernel_size = 7,
            stride = 1
        )
        
        ## Load SSL model
        self.modelname = modelname
        if modelname == 'hubert':
            model_id = "facebook/hubert-large-ls960-ft"
        elif modelname == 'w2v':
            model_id = "facebook/wav2vec2-large-960h-lv60-self"
        elif modelname == 'wavlm':
            model_id = "patrickvonplaten/wavlm-libri-clean-100h-large"
        else:
            raise ValueError("Error: [hubert, w2v, wavlm] required")
        
        self.ssl_model = AutoModel.from_pretrained(model_id)
        self.ssl_processor = AutoProcessor.from_pretrained(model_id)
            
        
        for param in self.ssl_model.parameters():
            param.requires_grad = False
        
                
        ## Linear Projection for HuBERT Embedding
        self.ssl_projection = Conv1d(
            in_channels = 1024,
            out_channels = min_dim*16//4,
            kernel_size = 1
        )
        
        self.downsampling_factor = 320  #2*4*5*8
        # 이거만 수정하면 될 듯
        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim*2, 2),
                                    EncBlock(min_dim*4, 4),
                                    EncBlock(min_dim*8, 5),
                                    EncBlock(min_dim*16, 8)                                        
                                    ])
        
        self.conv_bottle1 = Conv1d(
                                in_channels=min_dim*16,
                                out_channels = min_dim*16//4,
                                kernel_size = 7, 
                                stride = 1,
                                )
                                 # 위에서 concat 한 후 아래는 2로
        self.conv_bottle2 = Conv1d(
                                in_channels=(min_dim*16//4)*2,
                                out_channels = min_dim*16,
                                kernel_size = 7,
                                stride = 1,
                                )        
        
        self.decoder = nn.ModuleList([
                                    DecBlock(min_dim*8, 8),
                                    DecBlock(min_dim*4, 5),
                                    DecBlock(min_dim*2, 4),
                                    DecBlock(min_dim, 2),
                                    ])
        
        self.conv_out = Conv1d(
            in_channels = min_dim,
            out_channels = 1,
            kernel_size = 7,
            stride = 1,
        )
        
    def forward(self, x, HR):
        
        input = x
        #################### Length Adjustment
        ## x and HR has same shape
        ## Match into multiple of downsampling factor
        fragment = torch.randn(0).to(x.device)
        # print(fragment.shape,"shape frag")
        
        if x.dim()== 3: # N x 1 x L
            sig_len = x.shape[2]
            if sig_len % self.downsampling_factor != 0:
                new_len = sig_len // self.downsampling_factor * self.downsampling_factor
                fragment = x[:,:,new_len:].clone().to(x.device)  # 수정된 부분
                # fragment = x[:,:,sig_len:]
                x = x[:,:,:sig_len]
                HR = HR[:,:,:sig_len]
                
        if x.dim()==2:
            sig_len = x.shape[1]
            if sig_len % self.downsampling_factor != 0:
                new_len = sig_len // self.downsampling_factor * self.downsampling_factor
                fragment = x[:,new_len:].clone().to(x.device)  # 수정된 부분
                # fragment = x[:,sig_len:]
                x = x[:,:sig_len]
                HR = HR[:,:sig_len]
                
        while len(x.size()) < 3:
            x = x.unsqueeze(-2)
            HR = HR.unsqueeze(-2)
            # fragment = fragment.unsqueeze(-2).to(x.device)
            
        # print("Input Signal Length: ",sig_len, fragment.shape)
        
        if sig_len % self.downsampling_factor != 0:
            sig_len = sig_len // self.downsampling_factor * self.downsampling_factor
            x = x[:,:,:sig_len]
            HR = HR[:,:,:sig_len]
        # print("Input Signal Length: ",sig_len, fragment.shape)
        #################### Length Adjustment End


        skip = [x]
        
        x = self.conv_in(x)
        skip.append(x)

        for encoder in self.encoder:
            x = encoder(x)
            skip.append(x)

        x = self.conv_bottle1(x)
        # print("BottleNeck:",x.shape) 
        
        ######################## Extract HuBERT Embeddings: B x L x Dim
        
        # print("input hr shape before squeeze: ",input.shape)
        embedding = HuBERT_layer(self.ssl_model, 
                                 self.ssl_processor, 
                                 input.squeeze(1),
                                 modelname=self.modelname
                                 ).detach()

        # print(type(embedding))
        # print("Embedding shape Before Perm: ",torch.tensor(embedding).shape)
        # embedding = embedding.permute(0,2,1).detach()

        ## Embedding into 2dim
        # print(embedding.shape)
        embedding_reshape = embedding.reshape(-1, 1024)
        
        # Kmeans
        cluster_labels = self.kmeans.predict(embedding_reshape.detach().cpu().numpy())  # GPU 사용시, CPU로 이동
        quantized_embedding = self.kmeans.cluster_centers_[cluster_labels]
        # print("Quantized Embedding shape: ", quantized_embedding.shape)
        
        embedding_new = quantized_embedding.reshape(embedding.shape[0], embedding.shape[1], -1)
        embedding_new = torch.from_numpy(embedding_new)
        
        ## KMeans forward를 하기 위해 numpy array로 변환했다가 다시 Tensor로 변환
        # print(f"Quantized Embedding shape {embedding_new.shape} and type {type(embedding_new)}")
        
        dev = next(self.ssl_projection.parameters()).device
        hubert_embedding = self.ssl_projection(embedding_new.permute(0,2,1).to(dev))
        
        # print(f"Projected Embedding Shape: {embedding_new.shape} --> {hubert_embedding.shape}")
        ##############################
        
        # print("HuBERT:", hubert_embedding.shape)
        # print(x.device(), hubert_embedding.device())
        x = torch.cat((x.to(dev),hubert_embedding), dim=1)
        # print("Concat:", x.shape)
        x = self.conv_bottle2(x)

        skip = skip[::-1]

        for l in range(len(self.decoder)):
            x = x + skip[l]
            x = self.decoder[l](x)

        x = x + skip[4]
        x = self.conv_out(x)
        
        x = x + skip[5]
        
        #################### Length Adjustment
        if len(fragment.size()) == 2:
            fragment = fragment.unsqueeze(-2)
            
        # print(x.shape, fragment.shape, "Two Shapes")
        x = torch.cat((x,fragment),dim=-1)
        # print("Output Signal Length: ",x.size()[-1])
        
        return x