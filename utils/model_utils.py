
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, alexnet, AlexNet_Weights
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, PreTrainedModel, PretrainedConfig
import torch.nn as nn

# Define a custom truncated model
class TruncatedDeiT(nn.Module):
    def __init__(self, full_model, num_layers=10, from_above=True, encoder_only=False):
        super().__init__()
        self.mode=from_above
        self.encoder_only = encoder_only
        if from_above:
            self.embeddings = full_model.embeddings
            self.encoder = nn.ModuleList(full_model.encoder.layer[:num_layers])
            #self.layernorm = full_model.layernorm
        else:
            self.transformer = nn.ModuleList(full_model.encoder.layer[num_layers:])
            self.layernorm = full_model.layernorm
            self.pooler = full_model.pooler

    def forward(self, x):
        if self.mode:
            # Embedding
            embedding_output = self.embeddings(x)
            hidden_states = embedding_output

            # Pass through selected encoder layers
            for layer_module in self.encoder:
                hidden_states = layer_module(hidden_states)[0]

            # Final layer norm
            #hidden_states = self.layernorm(hidden_states)
        else:
            # Pass through selected encoder layers
            hidden_states = x
            for layer_module in self.transformer:
                hidden_states = layer_module(hidden_states)[0]
            if not self.encoder_only:
                hidden_states = self.layernorm(hidden_states)
                hidden_states = self.pooler(hidden_states)
        return hidden_states

class WrappedHuggingfaceModel(torch.nn.Module):
    def __init__(self, hugging_model):
        super().__init__()
        self.hugging_model = hugging_model

    def forward(self, pixel_values):
        outputs = self.hugging_model(pixel_values=pixel_values)
        return outputs.logits

class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.classification_head = nn.Linear(encoder.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        cls_token_output = outputs.last_hidden_state[:, 0, :]  # Extract the CLS token
        logits = self.classification_head(cls_token_output)
        return logits

# Define the classifier model
class FullClassifier(nn.Module):
    def __init__(self, encoder, strategy, mlp, pooled=False):
        super(FullClassifier, self).__init__()
        self.encoder = encoder
        self.strategy = strategy
        self.classification_head = mlp
        self.pooled = pooled
    def forward(self, x):
        outputs = self.encoder(x)
        #cls_token_output = outputs.last_hidden_state[:, 0, :]  # Extract the CLS token
        if self.pooled:
            output=outputs
        else:
            if self.strategy == 'cls':
                #print(outputs.shape)
                output = outputs[:, 0, :]
            elif self.strategy == 'mean':
                output = outputs.mean(dim=1)
        logits = self.classification_head(output)
        return logits

class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomMLP, self).__init__()
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CustomTransformer(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomTransformer, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_sizes[0], nhead=8),
            num_layers=len(hidden_sizes) - 1
        )
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        return x

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
def get_trocr(name='trocr-small-stage1', strategy='cls', mlp=CustomMLP(input_size=384, hidden_sizes=[128], output_size=2),pooled=True):
    if name == 'trocr-small-stage1':
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1')
        model = WrappedHuggingfaceModel(model.encoder) #remove pixel_values argument
    elif name == 'trocr-2transformer-blocks':
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1')
        model = TruncatedDeiT(model.encoder, num_layers=10, from_above=False, encoder_only=not(pooled))
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['trocr-small-stage1']")
    model=FullClassifier(model, strategy=strategy, mlp=mlp, pooled=pooled)
    return model

def get_model(name="resnet50", pretrained=True, num_classes=None, input_size=768, hidden_sizes=[512, 256]):
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
    elif name in ["trocr-small-stage1",'trocr-2transformer-blocks']:
        return get_trocr(name, strategy='cls', mlp=CustomMLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=2), pooled=True)
    elif name == "MLP":
        return CustomMLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=2)
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
    elif name=='MLP':
        return -1
    elif depth == -1:
        return -1
    #if -1 is returned all layers are trainable
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['resnet18', 'resnet50', 'efficientnet']")
