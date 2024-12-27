from utils import SileroVadDataset, SileroVadPadder, VADDecoderRNNJIT, train, validate, init_jit_model, predict
from omegaconf import OmegaConf
import torch.nn as nn
import torch



if __name__ == '__main__':
    config = OmegaConf.load('config.yml')

    val_dataset = SileroVadDataset(config, mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             collate_fn=SileroVadPadder,
                                             num_workers=config.num_workers,
                                             persistent_workers=True)

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
    model.to(config.device)
    decoder = VADDecoderRNNJIT().to(config.device)
    decoder.load_state_dict(model._model_8k.decoder.state_dict() if config.tune_8k else model._model.decoder.state_dict())
    decoder.train()
    params = decoder.parameters()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                 lr=config.learning_rate)
    criterion = nn.BCELoss(reduction='none')

    best_val_roc = 0
    val_loss, val_roc = validate(config, val_loader, model, decoder, criterion, config.device)
    
    print(f'\tValidation loss: {round(val_loss, 3)}\n'
          f'\tValidation ROC-AUC: {round(val_roc, 3)}')
    

    ################################################## Start cap nhat 19/11/2024
    print('Making predicts...')
    all_predicts, all_gts = predict(model, val_loader, config.device, sr=8000 if config.tune_8k else 16000)
    print('Calculating thresholds...')
    best_ths_enter, best_ths_exit, best_acc = calculate_best_thresholds(all_predicts, all_gts)
    print(f'Best threshold: {best_ths_enter}\nBest exit threshold: {best_ths_exit}\nBest accuracy: {best_acc}')
    ################################################# End cap nhat 19/11/2024
    
    # print(predict(model,val_loader,config.device,16000))
    input("Press Enter to continue...")



