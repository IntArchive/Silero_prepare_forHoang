from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import torchaudio
import warnings
import random
import torch
import gc





warnings.filterwarnings('ignore')


def read_audio(path: str,
               sampling_rate: int = 16000,
               normalize=False):

    wav, sr = torchaudio.load(path) #

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sampling_rate:
        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                       new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate

    if normalize and wav.abs().max() != 0:
        wav = wav / wav.abs().max()

    return wav.squeeze(0) #Tensor with all specified dimensions of input of size 1 removed.


def build_audiomentations_augs(config, p):
    '''
    The are two ways to apply augmentations. The first one always has AddBackgroundNoise and two others.
    The second one has 3 random transforms excluded AddBackgroundNoise

    :param noise_path: folder path to all files noise
    :param p: probability to apply augmentations which was defined in config.yml
    '''
    from torch_audiomentations import AddBackgroundNoise, BandPassFilter, BandStopFilter, LowPassFilter, AddColoredNoise, ApplyImpulseResponse, HighPassFilter, SomeOf
    

    prob_add_bg_noise = 0.8
    transform_add_noise = [AddBackgroundNoise(background_paths=config.noise_root_path,min_snr_in_db = -5,max_snr_in_db=35, p=1),
                           ApplyImpulseResponse(ir_paths=config.rir_root_path,convolve_mode="same", compensate_for_propagation_delay=True)]
    num_transfrom_after_addnoise = random.randint(1, 2)


    transforms = [BandPassFilter(p=1),
                  BandStopFilter(p=1),
                  LowPassFilter(min_cutoff_freq=3500, max_cutoff_freq=7500,p=1),
                  HighPassFilter(min_cutoff_freq=50, max_cutoff_freq=500,p=1),
                  AddColoredNoise(min_snr_in_db=-5, max_snr_in_db=35, min_f_decay=-1,max_f_decay=1)]
    

    # Beside add background noise, choose some other augments from list `transforms`
    start = random.randint(0, len(transforms) - num_transfrom_after_addnoise)
    end = start + num_transfrom_after_addnoise
    transform_add_noise.extend(transforms[start:end])


    tr = SomeOf(2 + num_transfrom_after_addnoise, transforms = transform_add_noise, p=p) if random.random() < prob_add_bg_noise else SomeOf((1, 3), transforms=transforms, p=p)
    return tr



class SileroVadDataset(Dataset):
    def __init__(self,
                 config,
                 mode='train'):

        self.num_samples = 512  # constant, do not change
        self.sr = 16000  # constant, do not change

        self.resample_to_8k = config.tune_8k
        self.noise_loss = config.noise_loss
        self.max_train_length_sec = config.max_train_length_sec
        self.max_train_length_samples = config.max_train_length_sec * self.sr

        assert self.max_train_length_samples % self.num_samples == 0
        assert mode in ['train', 'val']

        dataset_path = config.train_dataset_path if mode == 'train' else config.val_dataset_path
        self.dataframe = pd.read_feather(dataset_path).reset_index(drop=True)
        self.index_dict = self.dataframe.to_dict('index')
        self.mode = mode
        print(f'DATASET SIZE : {len(self.dataframe)}')

        if mode == 'train':
            self.augs = build_audiomentations_augs(config, p=config.aug_prob)
        else:
            self.augs = None

    def __getitem__(self, idx):
        idx = None if self.mode == 'train' else idx
        wav, gt, mask = self.load_speech_sample(idx)

        if self.mode == 'train':
            wav = torch.reshape(torch.from_numpy(wav),(1,1,-1))
            wav = self.add_augs(wav)
            len_wav = wav.size()[-1]

            if len_wav > self.max_train_length_samples:
                wav = wav[...,:self.max_train_length_samples]
                gt = gt[:int(self.max_train_length_samples / self.num_samples)]
                mask = mask[:int(self.max_train_length_samples / self.num_samples)]
            
            if self.resample_to_8k:
                transform = torchaudio.transforms.Resample(orig_freq=self.sr,
                                                           new_freq=8000)
                wav = transform(wav)
            return torch.squeeze(wav), torch.FloatTensor(gt), torch.from_numpy(mask)
        else:
            return torch.from_numpy(wav), torch.FloatTensor(gt), torch.from_numpy(mask)

    def __len__(self):
        return len(self.index_dict)

    def load_speech_sample(self, idx=None):
        if idx is None:
            idx = random.randint(0, len(self.index_dict) - 1)
        wav = read_audio(self.index_dict[idx]['audio_path'], self.sr).detach().cpu().numpy()

        if len(wav) % self.num_samples != 0:
            pad_num = self.num_samples - (len(wav) % self.num_samples)
            wav = np.pad(wav, (0, pad_num), 'constant', constant_values=0)


        gt, mask = self.get_ground_truth_annotated(self.index_dict[idx]['speech_ts'], len(wav))

        assert len(gt) == len(wav) / self.num_samples

        mask[gt == 0]

        return wav, gt, mask

    def get_ground_truth_annotated(self, annotation, audio_length_samples):
        gt = np.zeros(audio_length_samples)
        
        for i in annotation:
            try:
                gt[int(i['start'] * self.sr): int(i['end'] * self.sr)] = 1
            except:
                print(i)
                raise ValueError
        squeezed_predicts = np.average(gt.reshape(-1, self.num_samples), axis=1)
        squeezed_predicts = (squeezed_predicts > 0.5).astype(int)
        mask = np.ones(len(squeezed_predicts))
        mask[squeezed_predicts == 0] = self.noise_loss
        return squeezed_predicts, mask

    def add_augs(self, wav):
        while True:
            try:
                wav_aug = self.augs(wav, self.sr)
                if torch.isnan(wav_aug.max()) or torch.isnan(wav_aug.min()):
                    return wav
                return wav_aug
            except Exception as e:
                continue


def SileroVadPadder(batch):
    wavs = [batch[i][0] for i in range(len(batch))]
    labels = [batch[i][1] for i in range(len(batch))]
    masks = [batch[i][2] for i in range(len(batch))]

    wavs = torch.nn.utils.rnn.pad_sequence(
        wavs, batch_first=True, padding_value=0)

    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0)

    masks = torch.nn.utils.rnn.pad_sequence(
        masks, batch_first=True, padding_value=0)

    return wavs, labels, masks


class VADDecoderRNNJIT(nn.Module):

    def __init__(self):
        super(VADDecoderRNNJIT, self).__init__()

        self.rnn = nn.LSTMCell(128, 128)
        self.decoder = nn.Sequential(nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.Conv1d(128, 1, kernel_size=1),
                                     nn.Sigmoid())

    def forward(self, x, state=torch.zeros(0)):
        x = x.squeeze(-1)
        if len(state):
            h, c = self.rnn(x, (state[0], state[1]))
        else:
            h, c = self.rnn(x)

        x = h.unsqueeze(-1).float()
        state = torch.stack([h, c])
        x = self.decoder(x)
        return x, state


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config,
          loader,
          jit_model,
          decoder,
          criterion,
          optimizer,
          device, 
          writer,
          epoch_count_from_0):

    losses = AverageMeter()
    decoder.train()

    context_size = 32 if config.tune_8k else 64
    num_samples = 256 if config.tune_8k else 512
    stft_layer = jit_model._model_8k.stft if config.tune_8k else jit_model._model.stft
    encoder_layer = jit_model._model_8k.encoder if config.tune_8k else jit_model._model.encoder

    with torch.enable_grad():
        for _, (x, targets, masks) in tqdm(enumerate(loader), total=len(loader)):
            targets = targets.to(device)
            x = x.to(device)
            masks = masks.to(device)
            x = torch.nn.functional.pad(x, (context_size, 0))

            outs = []
            state = torch.zeros(0)
            for i in range(context_size, x.shape[1], num_samples):
                input_ = x[:, i-context_size:i+num_samples]
                out = stft_layer(input_)
                out = encoder_layer(out)
                out, state = decoder(out, state)
                outs.append(out)
            stacked = torch.cat(outs, dim=2).squeeze(1)

            loss = criterion(stacked, targets)
            loss = (loss * masks).mean()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), masks.numel())



            # Log every 1000 steps
            step = len(loader)*epoch_count_from_0 + _
            if step % 1000 == 0:
                writer.add_scalar(f'Loss/1k_step', losses.avg, _)

    writer.flush() # Ensure writer pending the ca

    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg


def validate(config,
             loader,
             jit_model,
             decoder,
             criterion,
             device):

    losses = AverageMeter()
    decoder.eval()

    predicts = []
    gts = []

    context_size = 32 if config.tune_8k else 64
    num_samples = 256 if config.tune_8k else 512
    stft_layer = jit_model._model_8k.stft if config.tune_8k else jit_model._model.stft
    encoder_layer = jit_model._model_8k.encoder if config.tune_8k else jit_model._model.encoder

    with torch.no_grad():
        for _, (x, targets, masks) in tqdm(enumerate(loader, start=1), total=len(loader)):
            targets = targets.to(device)
            x = x.to(device)
            masks = masks.to(device)
            x = torch.nn.functional.pad(x, (context_size, 0))

            outs = []
            state = torch.zeros(0)
            for i in range(context_size, x.shape[1], num_samples):
                input_ = x[:, i-context_size:i+num_samples]
                out = stft_layer(input_)
                out = encoder_layer(out)
                out, state = decoder(out, state)
                outs.append(out)
            stacked = torch.cat(outs, dim=2).squeeze(1)

            predicts.extend(stacked[masks != 0].tolist())
            gts.extend(targets[masks != 0].tolist())

            loss = criterion(stacked, targets)
            loss = (loss * masks).mean()
            losses.update(loss.item(), masks.numel())
    score = roc_auc_score(gts, predicts)

    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg, round(score, 3)


def init_jit_model(model_path: str,
                   device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def predict(model, loader, device, sr):
    with torch.no_grad():
        all_predicts = []
        all_gts = []
        for _, (x, targets, masks) in tqdm(enumerate(loader), total=len(loader)):
            x = x.to(device)
            out = model.audio_forward(x, sr=sr)

            for i, out_chunk in enumerate(out):
                predict = out_chunk[masks[i] != 0].cpu().tolist()
                gt = targets[i, masks[i] != 0].cpu().tolist()

                all_predicts.append(predict)
                all_gts.append(gt)
    return all_predicts, all_gts


def calculate_best_thresholds(all_predicts, all_gts):
    best_acc = 0
    for ths_enter in tqdm(np.linspace(0, 1, 20)):
        for ths_exit in np.linspace(0, 1, 20):
            if ths_exit >= ths_enter:
                continue

            accs = []
            for j, predict in enumerate(all_predicts):
                predict_bool = []
                is_speech = False
                for i in predict:
                    if i >= ths_enter:
                        is_speech = True
                        predict_bool.append(1)
                    elif i <= ths_exit:
                        is_speech = False
                        predict_bool.append(0)
                    else:
                        val = 1 if is_speech else 0
                        predict_bool.append(val)

                score = round(accuracy_score(all_gts[j], predict_bool), 4)
                accs.append(score)

            mean_acc = round(np.mean(accs), 3)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_ths_enter = round(ths_enter, 2)
                best_ths_exit = round(ths_exit, 2)
    return best_ths_enter, best_ths_exit, best_acc


def record_loss_after_one_thousand_step(train_loss, ):
    # Log the loss to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch * len(train_loader))
    writer.flush()
