
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, alexnet, AlexNet_Weights
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights
import torch

def get_efficientnet(pretrained=True, num_classes=None):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    if num_classes:
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    return model
def get_alexnet(pretrained=True, num_classes=None):
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = alexnet(weights=weights)
    if num_classes:
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes)
    return model
def get_resnet50(pretrained=True, num_classes=None):
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    if num_classes:
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    return model, weights
def get_vgg16(pretrained=True, num_classes=None):   
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg16(weights=weights)
    if num_classes:
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes)
    return model
def get_resnet18(pretrained=True, num_classes=None):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if num_classes:
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def get_model(name="resnet50", pretrained=True, num_classes=None):
    if name == "efficientnet":
        return get_efficientnet(pretrained, num_classes)
    elif name == "resnet50":
        return get_resnet50(pretrained, num_classes)
    elif name == "resnet18":
        return get_resnet18(pretrained, num_classes)
    elif name == "alexnet":
        return get_alexnet(pretrained, num_classes)
    elif name == "vgg16":
        return get_vgg16(pretrained, num_classes)
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['efficientnet', 'resnet50', 'resnet18', 'alexnet', 'vgg16']")

def get_weights(name="resnet50"):
    if name == "efficientnet":
        return EfficientNet_V2_S_Weights.IMAGENET1K_V1
    elif name == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V1
    elif name == "resnet18":
        return ResNet18_Weights.IMAGENET1K_V1
    elif name == "alexnet":
        return AlexNet_Weights.IMAGENET1K_V1
    elif name == "vgg16":
        return VGG16_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['efficientnet', 'resnet50', 'resnet18', 'alexnet', 'vgg16']")

def get_trainable_layers(name,depth=0):
    #this gives the number of layers to fine tune according to which part of the model 
    #you want to unfreeze (eg only classification layer, last convolutional layer, two convolutional layers)
    if depth == 0:
        return 1
    if name == 'resnet18':
        if depth == 1: #first convolutional layer
            return 4 #to check
        elif depth == 2: #last two convolutional layer
            return 9
    elif name == 'resnet50':
        if depth == 1: #first convolutional layer
            return 
        elif depth == 2: #last two convolutional layer
            return 
    elif name == 'efficientnet':
        if depth == 1: #first convolutional layer
            return 
        elif depth == 2: #last two convolutional layer
            return 
    elif name=='trocr-small-stage1':
        if depth == 1:
            return 16
        elif depth == 2:
            return 
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['resnet18', 'resnet50', 'efficientnet']")

class WrappedHuggingfaceModel(torch.nn.Module):
    def __init__(self, hugging_model):
        super().__init__()
        self.hugging_model = hugging_model

    def forward(self, pixel_values):
        outputs = self.hugging_model(pixel_values=pixel_values)
        return outputs.logits