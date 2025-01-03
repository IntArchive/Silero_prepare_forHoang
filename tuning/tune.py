from utils2 import SileroVadDataset, SileroVadPadder, VADDecoderRNNJIT, train, validate, init_jit_model, get_checkpoint
from omegaconf import OmegaConf
import torch.nn as nn
import torch
import profile
# Record SummaryWriter 
from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':
    config = OmegaConf.load('config.yml')

    mode = get_checkpoint()
    checkpoint_path = config.checkpoint_path

    train_dataset = SileroVadDataset(config, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               collate_fn=SileroVadPadder,
                                               num_workers=config.num_workers,
                                               persistent_workers=True)
    val_dataset = SileroVadDataset(config, mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             collate_fn=SileroVadPadder,
                                             num_workers=config.num_workers,
                                             persistent_workers=True)
    
    step_of_epoch = 0
    save_interval = 0
    if mode:
        if config.jit_model_path:
            print(f'Loading model from the local folder: {config.jit_model_path}')
            model = init_jit_model(config.jit_model_path, device=config.device)
        else:
            if config.use_torchhub:
                print('Loading model using torch.hub')
                model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        onnx=False,
                                        force_reload=True)
            else:
                print('Loading model using silero-vad library')
                from silero_vad import load_silero_vad
                model = load_silero_vad(onnx=False)
        

        print('Model loaded')
        checkpoint = torch.load(checkpoint_path)
        #model
        model.load_state_dict(checkpoint['model_state_dict']) #load_state_dict of the model
        model.to(config.device)

        #decoder
        decoder = VADDecoderRNNJIT().to(config.device)
        decoder.load_state_dict(model._model_8k.decoder.state_dict() if config.tune_8k else model._model.decoder.state_dict())   #we can reuse this line to load_state_dict
        decoder.train()
        params = decoder.parameters()


        #optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                    lr=config.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #criterion
        criterion = nn.BCELoss(reduction='none')
        
        #best_val_roc
        best_val_roc = checkpoint['best_val_roc']

        #pending epoch and step_of_epoch
        start_epoch = checkpoint['start_epoch'] if mode else 0
        step_of_epoch = checkpoint['step_of_epoch']

        save_interval = config.save_interval

    else:
        if config.jit_model_path:
            print(f'Loading model from the local folder: {config.jit_model_path}')
            model = init_jit_model(config.jit_model_path, device=config.device)
        else:
            if config.use_torchhub:
                print('Loading model using torch.hub')
                model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        onnx=False,
                                        force_reload=True)
            else:
                print('Loading model using silero-vad library')
                from silero_vad import load_silero_vad
                model = load_silero_vad(onnx=False)
        

        print('Model loaded')
        #Don't have checkpoint
        model.to(config.device)

        decoder = VADDecoderRNNJIT().to(config.device)
        decoder.load_state_dict(model._model_8k.decoder.state_dict() if config.tune_8k else model._model.decoder.state_dict())
        decoder.train()
        params = decoder.parameters()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                    lr=config.learning_rate)
        criterion = nn.BCELoss(reduction='none')

        best_val_roc = 0

        start_epoch = 0
        step_of_epoch = 0

        save_interval = config.save_interval


    writer = SummaryWriter(config.logdir_tensorboard)  # Logs are saved in 'runs/...'
    for i in range(start_epoch,config.num_epochs,1):
        start_epoch = i
        epoch_count_from_0 = i

        print(f'Starting epoch {i + 1}')
        train_loss = train(config, train_loader, model, decoder, criterion, optimizer, config.device, step_of_epoch, writer, epoch_count_from_0, best_val_roc, start_epoch, checkpoint_path, save_interval=save_interval)
        val_loss, val_roc = validate(config, val_loader, model, decoder, criterion, config.device)
        print(f'Metrics after epoch {i + 1}:\n'
              f'\tTrain loss: {round(train_loss, 3)}\n',
              f'\tValidation loss: {round(val_loss, 3)}\n'
              f'\tValidation ROC-AUC: {round(val_roc, 3)}')
        writer.add_scalar('TrainLoss/epoch', round(train_loss, 3), i + 1)
        writer.add_scalar('ValLoss/epoch', round(val_loss, 3), i + 1)
        writer.add_scalar('Validation ROC-AUC',round(val_roc, 3), i + 1)

        if val_roc > best_val_roc:
            print('New best ROC-AUC, saving model')
            best_val_roc = val_roc
            if config.tune_8k:
                model._model_8k.decoder.load_state_dict(decoder.state_dict())
            else:
                model._model.decoder.load_state_dict(decoder.state_dict())
            torch.jit.save(model, config.model_save_path)
        
    # Close the writer
    writer.close()

    print('Done')

