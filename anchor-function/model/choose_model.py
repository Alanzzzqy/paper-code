from .DNN import myDNN
from .LSTM import myLSTM
from .GPT import myGPT
from .GPT_new import myGPT_new
from .GPT_without import myGPT_without

def get_model(args, device):
    if args.model == 'LSTM':
        model = myLSTM(args, device).to(device)
    elif args.model == 'GPT':
        model = myGPT(args, device).to(device)
    elif args.model == 'DNN':
        model = myDNN(args, device).to(device)
    elif args.model == 'GPT_without':
        model = myGPT_without(args, device).to(device)
    elif args.model == 'GPT_new':
        model = myGPT_new(args, device).to(device)

    return model