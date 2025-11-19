from typing_extensions import OrderedDict
import torch
from transformers import VisionEncoderDecoderModel, ViTModel, ViTForImageClassification
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler

source_path = 'C:\\Users\\andre\\VsCode\\PD related projects\\gender_detection'
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

class ContrastiveModel(nn.Module):
    """ResNet Backbone + Projection Head for SimCLR."""
    def __init__(self, model, in_features,projection_dim=128,hidden_dim=256,model_type='resnet'):
        super().__init__()
        if model_type == 'resnet':
            self.encoder = model
            self.encoder.fc = nn.Identity()  # Remove the classification head
        elif model_type == 'vit':
            self.encoder = model
            #self.encoder.heads = nn.Identity()  # Remove the classification head
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)

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
    def __init__(self, input_size, hidden_sizes, output_size, **kwargs):
        super(CustomMLP, self).__init__()
        layers = []
        activation = kwargs.get('activation', 'relu')
        dropout = kwargs.get('dropout', None)
        batchnorm = kwargs.get('batchnorm', False)
        with_input_norm = kwargs.get('with_input_norm', None)
        scale = kwargs.get('scale', 1.0)
        mean = kwargs.get('mean', 0.0)
        if with_input_norm is None:
            pass
        elif with_input_norm=='batch_norm':
            layers.append(nn.BatchNorm1d(input_size))
        elif with_input_norm=='dataset_norm':
            layers.append(FeatureNormalizer(mean,scale))
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            if dropout is not None:
                if isinstance(dropout, float):
                    layers.append(nn.Dropout(dropout))
                else:
                    raise ValueError("Dropout should be a float value.")
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

class Custom1DCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Custom1DCNN, self).__init__()
        layers = []
        in_channels = 1  # Assuming input is a single channel (e.g., grayscale)
        for hidden_size in hidden_sizes:
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = hidden_size
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels * (input_size // (2 ** len(hidden_sizes))), output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.unsqueeze(1))  # Add channel dimension

class CustomLogreg(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomLogreg, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class JoinedModels(nn.Module):
    def __init__(self, vision_model, classifier):
        super().__init__()
        self.vision_model = vision_model
        self.classifier = classifier

    def forward(self, x):
        features = self.vision_model(x)  # image -> features
        logits = self.classifier(features)  # features -> prediction
        return logits
class SKLearnLogRegWrapper(nn.Module):
    def __init__(self, sklearn_model):
        super().__init__()
        coef = torch.tensor(sklearn_model.coef_, dtype=torch.float32)
        intercept = torch.tensor(sklearn_model.intercept_, dtype=torch.float32)
        self.linear = nn.Linear(coef.shape[1], coef.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(coef)
            self.linear.bias.copy_(intercept)

    def forward(self, x):
        logits = self.linear(x)
        p1 = torch.sigmoid(logits)
        p0 = 1 - p1
        return torch.cat([p0, p1], dim=1)  # Shape: [B, 2]

def get_custom_cnn(name, mode, pretrained, **kwargs):
    #https://chatgpt.com/share/68d26a38-d0f0-8010-9891-dc0770d96251
    class TinyTextCNN(nn.Module):
        def __init__(self, out_features=384):
            super().__init__()
            # (in_ch, out_ch, k, s, p)
            def block(ic, oc):
                return nn.Sequential(OrderedDict([
                        (f"conv1", nn.Conv2d(ic, oc, 3, 1, 1, bias=False)),
                        (f"bn1",   nn.BatchNorm2d(oc)),
                        (f"relu1", nn.ReLU(inplace=True)),
                        (f"conv2", nn.Conv2d(oc, oc, 3, 1, 1, bias=False)),
                        (f"bn2",   nn.BatchNorm2d(oc)),   # <-- named as bn2
                        (f"relu2", nn.ReLU(inplace=True)),
                        (f"pool",  nn.MaxPool2d(2)),
                        (f"drop",  nn.Dropout2d(0.1)),
                    ])
                )

            self.features = nn.Sequential(
                block(3,   32),   # 224 -> 112
                block(32,  64),   # 112 -> 56
                block(64, 128),   # 56  -> 28
                block(128, 256),  # 28  -> 14
                block(256, out_features),  # 14  -> 7
            )

            self.avgpool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # [B, C]
            return x  
    class ConvBNReLU(nn.Module):
        def __init__(self, ic, oc, k=3, s=1):
            super().__init__()
            p = k // 2
            self.conv = nn.Conv2d(ic, oc, k, s, p, bias=False)
            self.bn   = nn.BatchNorm2d(oc)
            self.act  = nn.ReLU(inplace=True)
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class TinyCNN_XS(nn.Module):
        def __init__(self, p_drop=0.2):
            super().__init__()
            # 224→112→56→28→14→7
            ch = [24, 48, 96, 160]  # reduced channels
            self.stage0 = ConvBNReLU(3, ch[0], k=3, s=2)     # 224→112

            self.stage1 = nn.Sequential(
                ConvBNReLU(ch[0], ch[0]),
                ConvBNReLU(ch[0], ch[1], s=2),             # 112→56
            )
            self.stage2 = nn.Sequential(
                ConvBNReLU(ch[1], ch[1]),
                ConvBNReLU(ch[1], ch[2], s=2),             # 56→28
            )
            self.stage3 = nn.Sequential(
                ConvBNReLU(ch[2], ch[2]),
                ConvBNReLU(ch[2], ch[3], s=2),             # 28→14
            )
            self.stage4 = nn.Sequential(
                ConvBNReLU(ch[3], ch[3], s=2),             # 14→7
            )

            self.dropout = nn.Dropout(p_drop)
            self.pool    = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x = self.stage0(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.dropout(x)
            x = self.pool(x).squeeze(-1).squeeze(-1)                 
            return x
    out_features= 384
    #model = TinyTextCNN(out_features=out_features)
    model = TinyCNN_XS()
    contrastive = kwargs.get('contrastive', False)
    if contrastive:
        contrastive_model = ContrastiveModel(model, in_features=out_features, projection_dim=128)
        return contrastive_model
    return model


def get_resnet(name,mode, pretrained, **kwargs):
    from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet101, ResNet101_Weights
    if name=='resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
    elif name=='resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif name=='resnet34':
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
    elif name=='resnet101':
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet101(weights=weights)
    elif name in ['resnet34_layer1','resnet34_layer2','resnet34_layer3']:
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        full_model = resnet34(weights=weights)
        if name=='resnet34_layer1':
            layers = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.relu,
                full_model.maxpool,
                full_model.layer1
            )
        elif name=='resnet34_layer2':
            layers = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.relu,
                full_model.maxpool,
                full_model.layer1,
                full_model.layer2
            )
        elif name=='resnet34_layer3':
            layers = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.relu,
                full_model.maxpool,
                full_model.layer1,
                full_model.layer2,
                full_model.layer3
            )
        class WrappedResNet(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers
                self.gap = torch.nn.AdaptiveAvgPool2d(1)

            def forward(self, x):
                x = self.layers(x)
                x = self.gap(x)  # [B, C, 1, 1]
                x = torch.flatten(x, 1)
                return x
        model = WrappedResNet(layers)
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['resnet50', 'resnet18']")
    contrastive = kwargs.get('contrastive', False)
    if contrastive:
        in_features = model.fc.in_features
        contrastive_model = ContrastiveModel(model, in_features=in_features, projection_dim=128)
        return contrastive_model
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
            if name not in ['resnet34_layer1','resnet34_layer2','resnet34_layer3']:
                model.fc = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
def get_mobilenet(name, mode, pretrained, **kwargs):
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    if name == 'mobilenet_v3_large':
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
    elif name == 'mobilenet_v3_small':
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)

    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.classifier[1].in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.classifier[1] = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            model.classifier = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
def get_convnext(name, mode, pretrained, **kwargs):
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            x=self.model(x) 
            x = x.flatten(1)
            return x
    class ConvNeXtStageOut(nn.Module):
        """
        Returns the tensor after a chosen stage of model.features.
        stage='stem'   → after features[0]
        stage='s1'     → after features[1]   (56×56, C=192)
        stage='ds1'    → after features[2]   (downsample to 28×28, C=384)
        stage='s2'     → after features[3]
        stage='ds2'    → after features[4]   (14×14, C=768)
        stage='s3'     → after features[5]
        stage='ds3'    → after features[6]   ( 7×7, C=1536)
        stage='s4'     → after features[7]
        """
        _map = {
            "stem": 0, "s1": 1, "ds1": 2, "s2": 3,
            "ds2": 4, "s3": 5, "ds3": 6, "s4": 7,
        }
        def __init__(self, convnext: nn.Module, stage: str = "s3"):
            super().__init__()
            assert stage in self._map, f"stage must be one of {list(self._map)}"
            self.model = convnext
            self.idx = self._map[stage]
            self.gap = nn.AdaptiveAvgPool2d(1)  # size-agnostic

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # run through features sequentially until desired stage
            for i, m in enumerate(self.model.features):
                x = m(x)
                if i == self.idx:
                    x = self.gap(x)
                    x = x.flatten(1)
                    return x
            return x  # fallback

    if name.startswith('convnext_base'):
        from torchvision.models import ConvNeXt_Base_Weights, convnext_base
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_base(weights=weights)
    if name.startswith('convnext_small'):
        from torchvision.models import ConvNeXt_Small_Weights, convnext_small
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_small(weights=weights)
    elif name.startswith('convnext_large'):
        from torchvision.models import ConvNeXt_Large_Weights, convnext_large
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_large(weights=weights)
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.classifier[1].in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.classifier[1] = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            if 'inter' in name:
                model = ConvNeXtStageOut(model, stage="s2")
            else:
                model.classifier = torch.nn.Identity()
                model = WrappedModel(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
def get_vgg(name, mode, pretrained, **kwargs):
    from torchvision.models import vgg16, VGG16_Weights, vgg11, VGG11_Weights, vgg13, VGG13_Weights, vgg19, VGG19_Weights
    class VGG_GAP_512(nn.Module):
        def __init__(self, vgg):
            super().__init__()
            self.features = vgg.features
            self.gap = nn.AdaptiveAvgPool2d(1)  # size-agnostic
            self.flatten = nn.Flatten(1)

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)          # [B, 512, 1, 1]
            return self.flatten(x)   # [B, 512]
    if name == 'vgg11':
        weights = VGG11_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg11(weights=weights)
    elif name == 'vgg13':
        weights = VGG13_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg13(weights=weights)
    elif name in ['vgg16', 'vgg16_512']:
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg16(weights=weights)
    elif name == 'vgg19':
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg19(weights=weights)
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['vgg11', 'vgg13', 'vgg16', 'vgg19']")
    
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.classifier[6].in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.classifier[6] = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            if name == 'vgg16':
                model.classifier[6] = torch.nn.Identity()
            else: 
                model = VGG_GAP_512(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
def get_densenet(name, mode, pretrained, **kwargs):
    from torchvision.models import densenet121, DenseNet121_Weights
    if name == 'densenet121':
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
    elif name == 'densenet169':
        from torchvision.models import densenet169, DenseNet169_Weights
        weights = DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet169(weights=weights)
    elif name == 'densenet161':
        from torchvision.models import densenet161, DenseNet161_Weights
        weights = DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet161(weights=weights)
    elif name == 'densenet201':
        from torchvision.models import densenet201, DenseNet201_Weights
        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet201(weights=weights)
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.classifier.in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.classifier = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            model.classifier = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
def get_alexnet(name, mode, pretrained, **kwargs):
    from torchvision.models import alexnet, AlexNet_Weights
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = alexnet(weights=weights)
    class wrappedAlex(nn.Module):
        def __init__(self, vgg):
            super().__init__()
            self.features = vgg.features
            self.gap = nn.AdaptiveAvgPool2d(1)  # size-agnostic
            self.flatten = nn.Flatten(1)

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)          
            return self.flatten(x)
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.classifier[6].in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.classifier[6] = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            if name == 'alexnet':
                model.classifier[6] = torch.nn.Identity()
            elif name == 'alexnet_gap':
                model = wrappedAlex(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
def get_efficientnet(name, mode, pretrained, **kwargs):
    if name == 'efficientnet_v2_s':
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_s(weights=weights)
    elif name == 'efficientnet_v2_m':
        from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_m(weights=weights)
    elif name == 'efficientnet_v2_l':
        from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_l(weights=weights)
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.classifier[1].in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.classifier[1] = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            model.classifier = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
def get_googlenet(name, mode, pretrained, **kwargs):
    from torchvision.models import googlenet, GoogLeNet_Weights
    weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = googlenet(weights=weights)
    
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.fc.in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.fc = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            model.fc = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
def get_inceptionv3(name, mode, pretrained, **kwargs):
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    model = inception_v3(weights=weights)
    
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.fc.in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.fc = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            model.fc = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model
def get_regnet(name, mode, pretrained, **kwargs):
    if name == "regnet_y_3_2":
        from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights
        weights = RegNet_Y_3_2GF_Weights.IMAGENET1K_V1 if pretrained else None
        model = regnet_y_3_2gf(weights=weights)
    elif name == "regnet_y_32":
        from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights
        weights = RegNet_Y_32GF_Weights.IMAGENET1K_V1 if pretrained else None
        model = regnet_y_32gf(weights=weights)
    elif name == "regnet_y_128":
        from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights
        weights = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        model = regnet_y_128gf(weights=weights)
    elif name == "regnet_x_3_2":
        from torchvision.models import regnet_x_3_2gf, RegNet_X_3_2GF_Weights
        weights = RegNet_X_3_2GF_Weights.IMAGENET1K_V1 if pretrained else None
        model = regnet_x_3_2gf(weights=weights) 
    elif name == "regnet_x_32":
        from torchvision.models import regnet_x_32gf, RegNet_X_32GF_Weights
        weights = RegNet_X_32GF_Weights.IMAGENET1K_V1 if pretrained else None
        model = regnet_x_32gf(weights=weights)
    
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.fc.in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.fc = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            model.fc = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model

def get_maxvit(name, mode, pretrained, **kwargs):
    from torchvision.models import maxvit_t, MaxVit_T_Weights
    weights = MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = maxvit_t(weights=weights)
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.gap = torch.nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x=self.model(x)
            x = self.gap(x)             # [B, C, 1, 1]
            x = x.flatten(1)            # shape: [B, C]
            return x

    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.classifier[1].in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.classifier[1] = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            model.classifier = torch.nn.Identity()
            model = WrappedModel(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model

def get_dbnet(name, mode, pretrained, **kwargs):
    from doctr.models import db_resnet50
    model = db_resnet50(pretrained=pretrained)
    model=model.feat_extractor
    class WrappedModel(torch.nn.Module):
        def __init__(self, model,level=3):
            super().__init__()
            self.model = model
            self.gap = torch.nn.AdaptiveAvgPool2d(1)
            self.level=level

        def forward(self, x):
            out = self.model(x)
            deepest_feature= out[str(self.level)]  # shape: [1, 2048, 32, 32]
            pooled = self.gap(deepest_feature)  # shape: [1, 2048, 1, 1]
            # Flatten to shape: [1, 2048]
            feature_vector = pooled.flatten(1)
            return feature_vector
    if mode=='classification head':
        print('no support for classification head for dbnet')
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            if 'inter' in name:
                return WrappedModel(model,level=1)
            else:
                return WrappedModel(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
def get_vitstr(name, mode, pretrained, **kwargs):
    from doctr.models import vitstr_base,vitstr_small
    if name.startswith('vitstr_small'):
        model = vitstr_small(pretrained=pretrained)
    elif name.startswith('vitstr_base'):
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
            self.gap = torch.nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x=self.model(x)['features']
            x = self.gap(x)             # [B, C, 1, 1]
            x = x.flatten(1)            # shape: [B, C]
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
            self.gap = torch.nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x=self.model(x)
            x = self.gap(x)             # [B, C, 1, 1]
            x = x.flatten(1)            # shape: [B, C]
            return x
    class CRNNFeatStage(nn.Module):
        """Return CNN feature map after a given index in feat_extractor."""
        def __init__(self, crnn, end_index: int):
            super().__init__()
            self.backbone = nn.Sequential(*list(crnn.children())[:end_index+1])
            self.gap = torch.nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x = self.backbone(x)
            x = self.gap(x)             # [B, C, 1, 1]
            x = x.flatten(1)            # shape: [B, C]
            return x
    if mode=='classification head':
        print('no support for classification head for dbnet')
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            if 'inter' in name:
                return CRNNFeatStage(model, end_index=23)
            else:
                return WrappedModel(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
def crnn_mobilenet_v3_large(name, mode, pretrained, **kwargs):
    from doctr.models import crnn_mobilenet_v3_large
    model = crnn_mobilenet_v3_large(pretrained=pretrained)
    model=model.feat_extractor
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.gap = torch.nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x=self.model(x)
            x = self.gap(x)             # [B, C, 1, 1]
            x = x.flatten(1)            # shape: [B, C]
            return x
    class CRNNStage(nn.Module):
        def __init__(self, model, end_index):
            super().__init__()
            # copy first N layers of feat_extractor
            self.backbone = nn.Sequential(*list(model.children())[:end_index + 1])
            self.gap = torch.nn.AdaptiveAvgPool2d(1)
        def forward(self, x):
            x = self.backbone(x)
            x = self.gap(x)             # [B, C, 1, 1]
            x = x.flatten(1)            # shape: [B, C]
            return x
    if mode=='classification head':
        print('no support for classification head for dbnet')
    elif mode=='as is':
        pass
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            if 'inter' in name:
                return CRNNStage(model, end_index=8)
            else:
                return WrappedModel(model)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
def get_db_mobilenet(name, mode, pretrained, **kwargs):
    from doctr.models import db_mobilenet_v3_large
    model = db_mobilenet_v3_large(pretrained=pretrained)
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
def get_linknet(name, mode, pretrained, **kwargs):
    from doctr.models import linknet_resnet18,linknet_resnet50
    if name.startswith('linknet_resnet18'):
        model = linknet_resnet18(pretrained=pretrained)
    elif name.startswith('linknet_resnet50'):
        model = linknet_resnet50(pretrained=pretrained)
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

### transformer models ###
def get_trocr(name, mode, pretrained, **kwargs):
    if pretrained:
        if 'inter' in name:
            name_temp=name.replace('-inter','')
        else:
            name_temp=name
        model = VisionEncoderDecoderModel.from_pretrained(f'microsoft/{name_temp}')
        #remove pixel_values argument; returns the output of the last layernorm (no pooling) 1,578,384
    else:
        print("no support for loading model without pretrained weights")
    class TrOCRViTClsExtractor(nn.Module):
        """
        Extract the [CLS] token from the N-th ViT encoder layer (default: 12th).
        Works with:
        - transformers.VisionEncoderDecoderModel (uses .encoder)
        - transformers.ViTModel
        Assumes pixel_values are already preprocessed (resize + normalize).
        """
        def __init__(self, trocr_or_vit, layer_index: int = 12):
            super().__init__()
            # If it's a VisionEncoderDecoderModel, grab the vision encoder
            self.encoder = getattr(trocr_or_vit, "encoder", trocr_or_vit)
            #num_layers = self.encoder.config.num_hidden_layers
            #assert 1 <= layer_index <= num_layers, f"layer_index must be in [1, {num_layers}]"
            self.layer_index = layer_index

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # Run ONLY the vision encoder and request hidden states
            out = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
            # hidden_states[0] = embeddings (patch+pos+cls); hidden_states[k] = after k-th block
            hs = out.hidden_states[self.layer_index]    # [B, seq_len, hidden_dim]
            cls = hs[:, 0, :]                           # [CLS]
            return cls
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
            if 'inter' in name:
                model = TrOCRViTClsExtractor(model, layer_index=12)
            else:
                model = WrappedHuggingfaceModel(model.encoder)
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
            #model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1')
            #model = TruncatedDeiT(model.encoder, num_layers=10, from_above=False, encoder_only=not(pooled))
    return model
def get_vit(name, mode, pretrained, **kwargs):
    if name in ["vit-base-patch16-224-in21k", "vit-base-patch16-224","vit-huge-patch14-224-in21k",
                "vit-large-patch16-224-in21k","vit-base-patch32-224-in21k"]:
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
def get_deit(name, mode, pretrained, **kwargs):
    class WrappedModel(torch.nn.Module):
        def __init__(self, hugging_model):
            super().__init__()
            self.hugging_model = hugging_model

        def forward(self, pixel_values):
            outputs = self.hugging_model(pixel_values=pixel_values,output_attentions=True)
            return outputs.hidden_states[-1][:, 0, :]  # Return the CLS token output
    class DeiTClsExtractor(nn.Module):
        """
        Extract the [CLS] token from the N-th ViT/DeiT encoder layer (default: 12th).
        Works with:
        - transformers.ViTForImageClassification  (DeiT often uses this class)
        - transformers.ViTModel
        - transformers.DeiTModel (if you use distilled variants; see token_index)
        Assumes pixel_values are already resized & normalized.
        """
        def __init__(self, vit_or_classifier, layer_index: int = 6, token_index: int = 0):
            super().__init__()
            # If it's a classifier, grab the base ViT
            self.vit = getattr(vit_or_classifier, "vit", vit_or_classifier)
            self.layer_index = layer_index

            # Which token to read: 0 = CLS; for DeiT distilled models, 1 can be distillation token
            self.token_index = token_index

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            out = self.vit(pixel_values=pixel_values, output_hidden_states=True)
            # hidden_states[0]: embeddings (after patch+pos(+cls/distill))
            # hidden_states[k]: output after k-th transformer block
            hs = out.hidden_states[self.layer_index]        # [B, seq_len, hidden_dim]
            cls_like = hs[:, self.token_index, :]           # pick CLS (or distill) token
            return cls_like

    from transformers import DeiTForImageClassificationWithTeacher
    if pretrained:
        if name=="DeiT-Tiny-Dist":
            model = DeiTForImageClassificationWithTeacher.from_pretrained(f'facebook/deit-tiny-distilled-patch16-224',output_hidden_states=True)
        elif name=="DeiT-Base-Dist":
            model = DeiTForImageClassificationWithTeacher.from_pretrained(f'facebook/deit-base-distilled-patch16-224',output_hidden_states=True)
        elif name=="DeiT-Small-Dist":
            model = DeiTForImageClassificationWithTeacher.from_pretrained(f'facebook/deit-small-distilled-patch16-224',output_hidden_states=True)
        if "DeiT-Tiny" in name:
            model = ViTForImageClassification.from_pretrained(f'facebook/deit-tiny-patch16-224',output_hidden_states=True)
        elif "DeiT-Small" in name:
            model = ViTForImageClassification.from_pretrained(f'facebook/deit-small-patch16-224',output_hidden_states=True)
        elif "DeiT-Base" in name:
            model = ViTForImageClassification.from_pretrained(f'facebook/deit-base-patch16-224',output_hidden_states=True)
    else:
        print("no support for loading model without pretrained weights")
    if mode == 'classification head':
        raise ValueError("Classification head is not supported for DeiT models.")
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            if 'inter' in name:
                model = DeiTClsExtractor(model, layer_index=6, token_index=0)
            else:
                model = WrappedModel(model) 
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    contrastive = kwargs.get('contrastive', False)
    if contrastive:
        in_features = 192
        contrastive_model = ContrastiveModel(model, in_features=in_features,projection_dim=128)
    return model
def get_beit(name, mode, pretrained, **kwargs):
    from transformers import BeitForImageClassification
    class BEiTClsExtractor(nn.Module):
        """
        Extract the [CLS] token from the N-th BEiT encoder layer (default: 12th).
        Works with either:
        - transformers.BeitModel
        - transformers.BeitForImageClassification (uses .beit internally)
        Assumes pixel_values are already preprocessed (resize + normalize).
        """
        def __init__(self, beit_or_classifier_model, layer_index: int = 12):
            super().__init__()
            # If it's a classifier, reach the base vision model via .beit
            self.beit = getattr(beit_or_classifier_model, "beit", beit_or_classifier_model)
            #num_layers = self.beit.config.num_hidden_layers
            #assert 1 <= layer_index <= num_layers, f"layer_index must be in [1, {num_layers}]"
            self.layer_index = layer_index

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # Request hidden states so we can index the chosen layer
            out = self.beit(pixel_values=pixel_values, output_hidden_states=True)
            # hidden_states[0] = embeddings (patch+pos); hidden_states[k] = after k-th block
            hs = out.hidden_states[self.layer_index]     # [B, seq_len, hidden_dim]
            cls = hs[:, 0, :]                            # [CLS] token
            return cls
    class WrappedModel(torch.nn.Module):
        def __init__(self, hugging_model):
            super().__init__()
            self.hugging_model = hugging_model

        def forward(self, pixel_values):
            outputs = self.hugging_model(pixel_values=pixel_values,output_attentions=True)
            return outputs.hidden_states[-1][:, 0, :]  # Return the CLS token output
    if pretrained:
        if "BEiT-Large" in name:
            model = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-384',output_hidden_states=True)
        elif "BEiT-Base" in name:   
            model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-384', output_hidden_states=True)
    else:
        print("no support for loading model without pretrained weights")
    if mode == 'classification head':
        raise ValueError("Classification head is not supported for DeiT models.")
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            if 'inter' in name:
                model = BEiTClsExtractor(model, layer_index=12)
            else:
                model = WrappedModel(model) 
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    contrastive = kwargs.get('contrastive', False)
    if contrastive:
        in_features = 192
        contrastive_model = ContrastiveModel(model, in_features=in_features,projection_dim=128)
    return model
def get_clip_vit(name, mode, pretrained, **kwargs):
    from transformers import CLIPModel
    normalization=True if name=="clip-vit-large-patch14" else False
    if name=="clip-vit-large-patch14-un":
        name="clip-vit-large-patch14"
    class WrappedModelInter(nn.Module):
        """
        Works with either:
        - transformers.CLIPVisionModel
        - transformers.CLIPModel  (uses .vision_model internally)
        Returns [CLS] from the specified vision block.
        """
        def __init__(self, model, layer_index: int = 12):
            super().__init__()
            # If it's a full CLIPModel, grab the vision tower
            self.vision = getattr(model, "vision_model", model)
            num_layers = self.vision.config.num_hidden_layers
            assert 1 <= layer_index <= num_layers, f"layer_index must be in [1, {num_layers}]"
            self.layer_index = layer_index

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # Run ONLY the vision encoder; request hidden states
            out = self.vision(pixel_values=pixel_values, output_hidden_states=True)
            hs = out.hidden_states[self.layer_index]   # [B, seq_len, hidden_dim]
            cls = hs[:, 0, :]                          # [CLS]
            return cls
    class WrappedModel(torch.nn.Module):
        def __init__(self, model, type_of_output='cls',normalization=False):
            super().__init__()
            self.model = model
            self.type_of_output = type_of_output
            self.normalization = normalization

        def forward(self, x):
            image_features = self.model.get_image_features(x)
            # Normalize the features (optional but common)
            if self.normalization:
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features
    class WrappedVisionModelExpl(torch.nn.Module):
        def __init__(self, vision_model, type_of_output='cls', normalization=False):
            super().__init__()
            self.vision_model = vision_model
            self.type_of_output = type_of_output  # Can be 'cls' or 'mean'
            self.normalization = normalization

        def forward(self, x):
            # Forward pass through the vision model
            outputs = self.vision_model(pixel_values=x, output_attentions=True)

            # Choose how to handle the output
            last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

            if self.type_of_output == 'cls':
                image_features = last_hidden[:, 0]  # CLS token
            elif self.type_of_output == 'mean':
                image_features = last_hidden.mean(dim=1)  # Mean pooling
            if self.normalization:
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            return image_features
    if pretrained:
        if 'inter' in name:
            name_temp=name.replace('-inter','')
        else:
            name_temp=name
        model = CLIPModel.from_pretrained(f'openai/{name_temp}')
        #model = WrappedHuggingfaceModel(model.vision_model) 
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
        return model
    elif mode=='truncated':
        truncation=kwargs.get('truncation', 'remove head')
        if truncation=='remove head':
            if 'inter' in name:
                return WrappedModelInter(model, layer_index=12)
            else:
                return WrappedModel(model,'cls',normalization) #add an option for other ways of reading the output
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    elif mode=='exp':
        return WrappedVisionModelExpl(model.vision_model, type_of_output=kwargs.get('type_of_output', 'cls'),normalization=normalization) #add an option for other ways of reading the output
def get_swin(name, mode, pretrained, **kwargs):
    if name == "swin_t":
        from torchvision.models import swin_t, Swin_T_Weights
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
    elif name == "swin_s":
        from torchvision.models import swin_s, Swin_S_Weights
        weights = Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_s(weights=weights)
    elif name == "swin_b":
        from torchvision.models import swin_b, Swin_B_Weights
        weights = Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_b(weights=weights)
    if mode == 'classification head':
        num_classes = kwargs.get('num_classes', 2)
        hidden_sizes = kwargs.get('hidden_sizes', [128])
        in_features = model.head.in_features
        mlp = CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
        model.head = mlp
    elif mode == 'as is':
        pass
    elif mode == 'truncated':
        truncation = kwargs.get('truncation', 'remove head')
        if truncation == 'remove head':
            model.head = torch.nn.Identity()
        else:
            raise ValueError(f"Truncation {truncation} is not supported. Choose from ['remove head']")
    return model

def get_model(name="resnet50", mode='classification head', pretrained=True, **kwargs):
    ''' 
    - name: the name of the model to download/load 
    -mode: 1) classification_head (modifies the last layers of the loaded model and appends an mlp classifier to the model)
    2) as is (loads the model as is, without any modifications)
    3) truncated (truncates the model to a certain number of layers) so that it returns an hidden representation    - pretrained: whether to load the pretrained weights or not
    - '''
    ### CNN MODELS ###
    if name.startswith('resnet'):
        model = get_resnet(name,mode, pretrained, **kwargs)
    elif name.startswith('vgg'):
        model = get_vgg(name, mode, pretrained, **kwargs)
    elif name.startswith("alexnet"):
        model = get_alexnet(name, mode, pretrained, **kwargs)
    elif name == "googlenet":
        model = get_googlenet(name, mode, pretrained, **kwargs)
    elif name.startswith('mobilenet'):
        model = get_mobilenet(name, mode, pretrained, **kwargs)
    elif name.startswith('convnext'):
        model = get_convnext(name, mode, pretrained, **kwargs)
    elif name.startswith('densenet'):
        model = get_densenet(name, mode, pretrained, **kwargs)
    elif name.startswith('efficientnet'):
        model = get_efficientnet(name, mode, pretrained, **kwargs)
    elif name.startswith('inception'):
        model = get_inceptionv3(name, mode, pretrained, **kwargs) 
    elif name.startswith('regnet'):
        model = get_regnet(name, mode, pretrained, **kwargs)
    ### hybrid MODELS ###
    elif name.startswith('maxvit'):
        model = get_maxvit(name, mode, pretrained, **kwargs)
    ### OCR MODELS ###
    elif name.startswith("dresnet50"):
        model = get_dbnet(name, mode, pretrained, **kwargs)
    elif name.startswith('vitstr'):
        model = get_vitstr(name, mode, pretrained, **kwargs)
    elif name.startswith('sar'):
        model = get_sar_resnet31(name, mode, pretrained, **kwargs)
    elif name.startswith('crnn_vgg16_bn'):
        model = get_crnn_vgg16_bn(name, mode, pretrained, **kwargs)
    elif name == "db_mobilenet":
        model = get_db_mobilenet(name, mode, pretrained, **kwargs)
    elif name.startswith('linknet'):
        model = get_linknet(name, mode, pretrained, **kwargs)
    elif name.startswith('crnn_mobilenet'):
        model = crnn_mobilenet_v3_large(name, mode, pretrained, **kwargs)
    ### TRANSFORMER MODELS ###
    elif name.startswith('trocr'):
        model = get_trocr(name,mode, pretrained, **kwargs)
    elif name in ["vit-base-patch16-224-in21k", "vit-base-patch16-224","vit-huge-patch14-224-in21k",
                "vit-large-patch16-224-in21k","vit-base-patch32-224-in21k"]:
        model = get_vit(name, mode, pretrained, **kwargs)
    elif name == "layoutlmv3_base":
        model = get_layoutlmv3_base(name, mode, pretrained, **kwargs)
    elif name.startswith('clip-vit'):
        model = get_clip_vit(name, mode, pretrained, **kwargs)
    elif name.startswith('DeiT'):
        model = get_deit(name, mode, pretrained, **kwargs)
    elif name.startswith('BEiT'):
        model = get_beit(name, mode, pretrained, **kwargs)
    elif name.startswith('swin'):
        model = get_swin(name, mode, pretrained, **kwargs)
    ### CUSTOM MODELS ###
    elif name == "custom_cnn":
        model = get_custom_cnn(name, mode, pretrained, **kwargs)
    #num_classes=num_classes, hidden_sizes=hidden_sizes, strategy=kwargs.get('strategy', 'cls'), pooled=kwargs.get('pooled', True)    
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['resnet50', 'resnet18', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet', 'googlenet', 'trocr family', 'vit family', and others]")
    pretrained_modality = kwargs.get('custom_pretrained','original')
    return model

def get_classification_head(name='MLPClassifier1',in_features=512,num_classes=2,**kwargs):
    dropout = kwargs.get('dropout', None)
    activation = kwargs.get('activation', 'relu')
    n_neurons = kwargs.get('n_neurons', 128)
    with_input_norm = kwargs.get('with_input_norm', None)
    scale = kwargs.get('scale', 1.0)
    mean = kwargs.get('mean', 0.0)
    if name == 'MLPClassifier1': #1 hidden layer
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons]) 
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes,
                         dropout=dropout, activation=activation, with_input_norm=with_input_norm,scale=scale, mean=mean)
    elif name == 'MLPClassifier2':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons,n_neurons]) 
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes,
                         dropout=dropout, activation=activation,with_input_norm=with_input_norm,scale=scale, mean=mean)
    elif name == 'MLPClassifier3':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons,n_neurons,n_neurons])
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes,dropout=dropout, 
                         activation=activation,with_input_norm=with_input_norm,scale=scale, mean=mean)
    if name == 'MLPClassifier1-BatchNorm': #1 hidden layer
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons]) 
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes, activation='relu',batchnorm=True)
    elif name == 'MLPClassifier2-BatchNorm':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons,n_neurons])
        return CustomMLP(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes, dropout=0.2 ,activation='relu',batchnorm=True)
    elif name == 'TransformerClassifier':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons,n_neurons])
        return CustomTransformer(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
    elif name == '1DCNNClassifier':
        hidden_sizes = kwargs.get('hidden_sizes',[n_neurons])
        return Custom1DCNN(input_size=in_features, hidden_sizes=hidden_sizes, output_size=num_classes)
    elif name == 'logreg':
        return CustomLogreg(input_size=in_features, output_size=num_classes)
    else:
        raise ValueError(f"Classification head {name} is not supported. Choose from ['MLPClassifier1', 'MLPClassifier2', 'TransformerClassifier']")

def get_sklearn_model(name='logreg', **kwargs): 
    if name=='svm':
        from sklearn.svm import SVC
        return SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)
    elif name=='logreg':
        from sklearn.linear_model import LogisticRegression
        penalty=kwargs.get('penalty','l2')
        C=kwargs.get('C',1.0)
        solver=kwargs.get('solver','lbfgs')
        max_iter=kwargs.get('max_iter',5000)
        return LogisticRegression(max_iter=max_iter, random_state=42, penalty=penalty, C=C, solver=solver)
    elif name=='gbm':
        # Define the models
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=100, #100 is standard 
            learning_rate=0.1,  
            max_depth=3,  
            random_state=42
        )
    elif name=='lgbm':
        import lightgbm as lgb
        from lightgbm import early_stopping, log_evaluation
        return lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=20,
            min_child_samples=30,#Minimum number of data samples per leaf
            subsample=0.8, #Randomness in row 
            colsample_bytree=0.8, #and feature sampling respectively.
            reg_alpha=1.0, # L1 regularization
            reg_lambda=1.0, # L2 regularization
            random_state=42,
            n_jobs=-1,
            min_split_gain=0.01,  # Minimum gain to make a split
        )
    elif name=='xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    #rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    elif name=='rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200,            # More trees = more stable
            max_depth=10,                # Limits tree depth (main regularizer)
            min_samples_split=10,        # Minimum samples to split a node
            min_samples_leaf=5,          # Minimum samples at a leaf node
            max_features='sqrt',         # Random feature selection at each split
            bootstrap=True,              # Use bootstrapped samples (default)
            oob_score=True,              # Out-of-bag error estimate
            random_state=42,
            n_jobs=-1
        )
    elif name=='mlp':
        from sklearn.neural_network import MLPClassifier
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', 256)
        return MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), activation='relu', solver='adam',
                            max_iter=200, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
    elif name=='dt':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=2, ccp_alpha=0.01, random_state=42)

def get_weights(name="resnet50"):
    if name == "efficientnet":
        return EfficientNet_V2_S_Weights.IMAGENET1K_V1
    elif name == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V1
    elif name == "resnet18":
        return ResNet18_Weights.IMAGENET1K_V1
    elif name == "alexnet":
        return AlexNet_Weights.IMAGENET1K_V1
    elif name == "vgg11":
        return VGG11_Weights.IMAGENET1K_V1
    elif name == "vgg13":
        return VGG13_Weights.IMAGENET1K_V1
    elif name == "vgg16":
        return VGG16_Weights.IMAGENET1K_V1
    elif name == "vgg19":
        return VGG19_Weights.IMAGENET1K_V1
    elif name == "googlenet":
        return GoogLeNet_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['efficientnet', 'resnet50', 'resnet18', 'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'googlenet']")

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
    elif name in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        if depth == 1: #last convolutional layer
            return 4 #features layers to unfreeze
        elif depth == 2: #last two convolutional layers
            return 8
    elif name == 'alexnet':
        if depth == 1: #last convolutional layer
            return 4 #features layers to unfreeze
        elif depth == 2: #last two convolutional layers
            return 8
    elif name == 'googlenet':
        if depth == 1: #last inception block
            return 6
        elif depth == 2: #last two inception blocks
            return 12
    elif name == 'MLP':
        return -1
    elif depth == -1:
        return -1
    #if -1 is returned all layers are trainable    else:
        raise ValueError(f"Model {name} is not supported. Choose from ['resnet18', 'resnet50', 'efficientnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet', 'googlenet', 'MLP']")

def test_output(size,transform, model):
    dummy_input = torch.rand(1, 3, size, size)
    dummy_input.shape
    '''if huggingface:
        # the transform is actually an huggingface processor in this case
        inputs = transform(images=dummy_input, return_tensors="pt")
        # Remove batch dimension from inputs
        patch = inputs['pixel_values'].squeeze()
    else:
        patch = transform(dummy_input)'''
    with torch.no_grad():
        output = model(dummy_input)
    return output

def get_custom_pretrained_weights(name,model,**kwargs):
    pretrained_modality = kwargs['custom_pretrained']
    which_checkpoint = kwargs.get('which_checkpoint', 'last')
    if which_checkpoint == 'best':
        checkpoint = torch.load(source_path+f'\\outputs\\online_deep_feature_extraction\\{name}\\{pretrained_modality}\\checkpoints\\checkpoint_best.pt', map_location='cpu', weights_only=False)
    else:
        checkpoint = torch.load(source_path+f'\\outputs\\online_deep_feature_extraction\\{name}\\{pretrained_modality}\\checkpoints\\checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_param_names(model, type_of_model):
    if type_of_model == 'contrastive':
        backbone_name = 'encoder.'
        classifier_name = 'projection_head.'
    else:
        backbone_name = 'vision_model.'
        classifier_name = 'classifier.'
    all_param_names = [name for name, _ in model.named_parameters()]
    backbone_param_names = [name for name, _ in model.named_parameters() if name.startswith(backbone_name)]
    classifier_param_names = [name for name, _ in model.named_parameters() if name.startswith(classifier_name)]
    return all_param_names, backbone_param_names, classifier_param_names

class FeatureNormalizer(nn.Module):
    def __init__(self, mean, std, learnable=False):
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        self.mean = nn.Parameter(mean, requires_grad=learnable)
        self.std = nn.Parameter(std, requires_grad=learnable)

    def forward(self, x):
        return (x - self.mean) / self.std

def get_normalization_parameters(train_file_name):
    #train_df.sort_values(by='page', inplace=True)
    train_df = pd.read_csv(train_file_name)
    cols_to_keep = [c for c in train_df.columns if c.startswith('f') and len(c) > 1 and c[1].isdigit()]
    X_train = train_df[cols_to_keep].values
    scaler = StandardScaler().fit(X_train)
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_
    return scaler_mean, scaler_scale

def compute_output_gpu(model, device, batch):
    model.eval()
    with torch.no_grad():
        images = batch['image'].to(device)
        outputs = model(images)
    return outputs