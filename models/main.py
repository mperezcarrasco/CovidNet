import torch
import torch.nn as nn
import torchvision.models as models


def build_model(args, device):
    """ Builds the model architecture to be used for the experiment."""
    if args.model=='Resnet50':
        model = models.resnet50(pretrained=False).to(device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3).to(device)
        if args.train==False:
            model.load_state_dict(torch.load('{}/trained_parameters_{}.pth'.format(
                                  args.model_path, args.model)))
            return model
    elif args.model=='CovidNet':
        #TODO
        pass
    return model