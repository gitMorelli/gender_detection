
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, alexnet, AlexNet_Weights
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights
import torch
from transformers import VisionEncoderDecoderModel, ViTModel
import torch.nn as nn
import torch.nn.functional as F

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
        return outputs.last_hidden_state

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

def get_resnet(name,mode, pretrained, **kwargs):
    if name=='resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
    elif name=='resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['resnet50', 'resnet18']")
    if mode=='classification head':
        num_classes=kwargs.get('num_classes', 2)
        hidden_sizes=kwargs.get('hidden_sizes', [128])
        in_features = model.fc.in_features
        mlp=CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.fc = mlp
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            model.fc = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
'''def get_efficientnet(pretrained=True):
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
    return model'''
'''
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
'''
def get_trocr(name, mode, pretrained, **kwargs):
    if name in ['trocr-small-stage1','trocr-small-handwritten']:
        if pretrained:
            model = VisionEncoderDecoderModel.from_pretrained(f'microsoft/{name}')
            model = WrappedHuggingfaceModel(model.encoder) 
            #remove pixel_values argument; returns the output of the last layernorm (no pooling) 1,578,384
        else:
            print("no support for loading model without pretrained weights")
        if mode=='classification head':
            num_classes=kwargs.get('num_classes', 2)
            hidden_sizes=kwargs.get('hidden_sizes', [128])
            how_to_read=kwargs.get('how_to_read', 'cls')
            if how_to_read=='cls':
                in_features = 384
            else:
                print('still no support for pooling or other averaging techniques')
            mlp=CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
            
        elif mode=='as is':
            pass
        elif mode=='truncated':
            truncation=kwargs.get('truncation', 'remove head')
            if truncation=='remove head':
                pass #I simply take the output of the last encoder layer (as in the pretrained model)
            else:
                raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
                model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1')
                model = TruncatedDeiT(model.encoder, num_layers=10, from_above=False, encoder_only=not(pooled))
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['trocr-small-stage1','trocr-small-handwriting']")
    return model
def get_vit(name, mode, pretrained, **kwargs):
    if name in ["vit-base-patch16-224-in21k", "vit-base-patch16-224"]:
        if pretrained:
            model = ViTModel.from_pretrained(f'google/{name}')
            model = WrappedHuggingfaceModel(model) #output size is [1, 197, 768]
        else:
            print("no support for loading model without pretrained weights")
        if mode=='classification head':
            raise ValueError("Classification head is not supported for ViT models.")
        elif mode=='as is':
            pass
        elif mode=='truncated':
            truncation=kwargs.get('truncation', 'remove head')
            if truncation=='remove head':
                pass #I simply take the output of the last encoder layer (as in the pretrained model)
            else:
                raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['vit-base-patch16-224-in21k', 'vit-base-patch16-224']")
    return model
def get_dbnet(name, mode, pretrained, **kwargs):
    from doctr.models import db_resnet50
    model = db_resnet50(pretrained=pretrained)
    model=model.feat_extractor
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            deepest_feature= out['3']  # shape: [1, 2048, 32, 32]
            pooled = F.adaptive_avg_pool2d(deepest_feature, (1, 1))  # shape: [1, 2048, 1, 1]
            # Flatten to shape: [1, 2048]
            feature_vector = pooled.view(pooled.size(0), -1)
            return feature_vector
    if mode=='classification head':
        print('no support for classification head for dbnet')
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            return WrappedModel(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
def get_vitstr_base(name, mode, pretrained, **kwargs):
    from doctr.models import vitstr_base
    model = vitstr_base(pretrained=pretrained)
    model=model.feat_extractor
    class WrappedModel(torch.nn.Module):
        def __init__(self, model, type_of_output='cls'):
            super().__init__()
            self.model = model
            self.type_of_output = type_of_output

        def forward(self, x):
            out = self.model(x)
            x=out['features']  
            if self.type_of_output=='cls':
                x=x[:,0,:]
            else:
                pass
            return x
    if mode=='classification head':
        print('no support for classification head for dbnet')
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            return WrappedModel(model,'cls') #add an option for other ways of reading the output
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
def get_sar_resnet31(name, mode, pretrained, **kwargs):
    from doctr.models import sar_resnet31
    model = sar_resnet31(pretrained=pretrained)
    model=model.feat_extractor
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            x=out['features']  # shape: [1, 512, 32, 32]
            x = F.adaptive_avg_pool2d(x, (1, 1))  # [1, 512, 1, 1]
            x = x.view(x.size(0), -1)  # Flatten to shape: [1, 512]
            return x
    if mode=='classification head':
        print('no support for classification head for dbnet')
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            return WrappedModel(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
def get_crnn_vgg16_bn(name, mode, pretrained, **kwargs):
    from doctr.models import crnn_vgg16_bn
    model = crnn_vgg16_bn(pretrained=pretrained)
    model=model.feat_extractor
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            x=self.model(x)
            x = x.squeeze(2)           # [B, C, W]
            x = x.permute(0, 2, 1)     # [B, W, C]
            x = x.mean(dim=1)
            return x
    if mode=='classification head':
        print('no support for classification head for dbnet')
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            return WrappedModel(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
def get_layoutlmv3_base(name, mode, pretrained, **kwargs): #need to test
    from transformers import LayoutLMv3ForTokenClassification
    if pretrained:
        model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=... )
        model = WrappedHuggingfaceModel(model.encoder) 
        #remove pixel_values argument; returns the output of the last layernorm (no pooling) 1,578,384
    else:
        print("no support for loading model without pretrained weights")
    if mode=='classification head':
        num_classes=kwargs.get('num_classes', 2)
        hidden_sizes=kwargs.get('hidden_sizes', [128])
        how_to_read=kwargs.get('how_to_read', 'cls')
        if how_to_read=='cls':
            in_features = 384
        else:
            print('still no support for pooling or other averaging techniques')
        mlp=CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            pass #I simply take the output of the last encoder layer (as in the pretrained model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
            model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1')
            model = TruncatedDeiT(model.encoder, num_layers=10, from_above=False, encoder_only=not(pooled))

def get_model(name="resnet50", mode='classification head', pretrained=True, **kwargs):
    ''' 
    - name: the name of the model to download/load 
    -mode: 1) classification_head (modifies the last layers of the loaded model and appends an mlp classifier to the model)
    2) as is (loads the model as is, without any modifications)
    3) truncated (truncates the model to a certain number of layers) so that it returns an hidden representation
    - pretrained: whether to load the pretrained weights or not
    - '''
    if name in ["resnet50",'resnet18']:
        return get_resnet(name,mode, pretrained, **kwargs)
    elif name in ["trocr-small-stage1",'trocr-small-handwritten']:
        return get_trocr(name,mode, pretrained, **kwargs)
    elif name in ["vit-base-patch16-224-in21k", "vit-base-patch16-224"]:
        return get_vit(name, mode, pretrained, **kwargs)
    elif name == "dresnet50":
        return get_dbnet(name, mode, pretrained, **kwargs)
    elif name == "vitstr_base":
        return get_vitstr_base(name, mode, pretrained, **kwargs)
    elif name == "sar_resnet31":
        return get_sar_resnet31(name, mode, pretrained, **kwargs)
    elif name == "crnn_vgg16_bn":
        return get_crnn_vgg16_bn(name, mode, pretrained, **kwargs)
    elif name == "layoutlmv3_base":
        return get_layoutlmv3_base(name, mode, pretrained, **kwargs)
    #num_classes=num_classes, hidden_sizes=hidden_sizes, strategy=kwargs.get('strategy', 'cls'), pooled=kwargs.get('pooled', True)
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['resnet50', trocr family]")

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
