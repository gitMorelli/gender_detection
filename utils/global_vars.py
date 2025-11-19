
# models.py
from dataclasses import dataclass
from functools import lru_cache

@dataclass(frozen=True)
class ModelProps:
    hugging: bool #True, False
    library: str #torchvision, doctr, huggingface
    architecture: str #transformer, cnn, hybrid
    architecture_specific: str = ''#vit, swin, deit, beit, resnet, convnext, mobilenet, crnn, sar, vitstr, trocr, dresnet, dbnet  
    pretraining_dataset: str = ''#doctr,trocr,trocr+iam,imagenet1k,imagenet21k,Crawled
    pretraining_mode: str = ''#contrastive, masking, classification, text_recognition, text_detection
    exclude_from_fe: bool = False
    # If you have optional/unknown extras:
    # extras: dict[str, object] = field(default_factory=dict)
    #i can get the depth, the number of parameters, the input size, by querying the model itself
    #ImageNet-21k 14mln images
    #imagenet1k 1.2mln images
# Single shared mapping (module-level constant)
_MODELS: dict[str, ModelProps] = {
    ### transformer models ###
    "swin_b":   ModelProps(hugging=False, library="torchvision", architecture="transformer"
                           ,architecture_specific="swin-b",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "swin_s":   ModelProps(hugging=False, library="torchvision", architecture="transformer"
                           ,architecture_specific="swin-s",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "BEiT-Base":   ModelProps(hugging=True, library="huggingface", architecture="transformer"
                               ,architecture_specific="beit-b",pretraining_dataset="imagenet21k",pretraining_mode="masking"),

    "BEiT-Large":  ModelProps(hugging=True, library="huggingface", architecture="transformer"
                               ,architecture_specific="beit-l",pretraining_dataset="imagenet21k",pretraining_mode="masking"),

    "BEiT-Large-inter":  ModelProps(hugging=True, library="huggingface", architecture="transformer"
                                      ,architecture_specific="beit-l",pretraining_dataset="imagenet21k",pretraining_mode="masking"),

    "vit-base-patch16-224": ModelProps(hugging=True, library="huggingface", architecture="transformer"
                                        ,architecture_specific="vit-b/16",pretraining_dataset="imagenet21k",pretraining_mode="classification"),

    "vit-base-patch32-224-in21k": ModelProps(hugging=True, library="huggingface", architecture="transformer"
                                              ,architecture_specific="vit-b/32",pretraining_dataset="imagenet21k",pretraining_mode="classification"),

    "vit-large-patch16-224-in21k": ModelProps(hugging=True, library="huggingface", architecture="transformer"
                                               ,architecture_specific="vit-l/16",pretraining_dataset="imagenet21k",pretraining_mode="classification"),

    "vit-huge-patch14-224-in21k": ModelProps(hugging=True, library="huggingface", architecture="transformer", 
                                              architecture_specific="vit-h/14",pretraining_dataset="imagenet21k",pretraining_mode="classification"),

    "vit-base-patch16-224-in21k": ModelProps(hugging=True, library="huggingface", architecture="transformer", 
                                              architecture_specific="vit-b/16",pretraining_dataset="imagenet21k",pretraining_mode="classification"),

    "DeiT-Small":  ModelProps(hugging=True, library="huggingface", architecture="transformer"
                               ,architecture_specific="vit-s",pretraining_dataset="imagenet1k",pretraining_mode="classification+distillation"),

    "DeiT-Small-Dist": ModelProps(hugging=True, library="huggingface", architecture="transformer"
                                   ,architecture_specific="deit-s",pretraining_dataset="imagenet1k",pretraining_mode="classification+distillation"),

    "DeiT-Base":   ModelProps(hugging=True, library="huggingface", architecture="transformer"
                               ,architecture_specific="vit-b",pretraining_dataset="imagenet1k",pretraining_mode="classification+distillation"),

    "DeiT-Tiny":  ModelProps(hugging=True, library="huggingface", architecture="transformer", 
                              architecture_specific="vit-t",pretraining_dataset="imagenet1k",pretraining_mode="classification+distillation"),

    "DeiT-Tiny-inter":  ModelProps(hugging=True, library="huggingface", architecture="transformer", 
                                architecture_specific="vit-t",pretraining_dataset="imagenet1k",pretraining_mode="classification+distillation"),

    "clip-vit-base-patch16": ModelProps(hugging=True, library="huggingface", architecture="transformer", 
                                        architecture_specific="vit-b/16",pretraining_dataset="Clip-vit",pretraining_mode="contrastive"),

    "clip-vit-base-patch32": ModelProps(hugging=True, library="huggingface", architecture="transformer", 
                                        architecture_specific="vit-b/32",pretraining_dataset="Clip-vit",pretraining_mode="contrastive"),

    "clip-vit-large-patch14-un": ModelProps(hugging=True, library="huggingface", architecture="transformer", 
                                            architecture_specific="vit-l/14",pretraining_dataset="Clip-vit",pretraining_mode="contrastive"),

    "clip-vit-large-patch14": ModelProps(hugging=True, library="huggingface", architecture="transformer",exclude_from_fe=True, 
                                         architecture_specific="vit-l/14",pretraining_dataset="Clip-vit",pretraining_mode="contrastive"),

    "clip-vit-large-patch14-inter": ModelProps(hugging=True, library="huggingface", architecture="transformer", 
                                               architecture_specific="vit-l/14",pretraining_dataset="Clip-vit",pretraining_mode="contrastive"),

    ### ocr models ###
    ### recognition###
    "crnn_mobilenet": ModelProps(hugging=False, library="doctr", architecture="cnn",
                                 architecture_specific="mobilenet_v3_l", pretraining_dataset="doctr",pretraining_mode="text_recognition",exclude_from_fe=True),
    
    "crnn_mobilenet_224": ModelProps(hugging=False, library="doctr", architecture="cnn", 
                                     architecture_specific="mobilenet_v3_l", pretraining_dataset="doctr",pretraining_mode="text_recognition"),
    
    "crnn_mobilenet_224-inter": ModelProps(hugging=False, library="doctr", architecture="cnn", 
                                          architecture_specific="mobilenet_v3_l", pretraining_dataset="doctr",pretraining_mode="text_recognition"),
    
    "sar_resnet31": ModelProps(hugging=False, library="doctr", architecture="cnn",
                               architecture_specific='resnet31', pretraining_dataset="doctr", pretraining_mode="text_recognition", exclude_from_fe=True),
    
    "sar_resnet31_224": ModelProps(hugging=False, library="doctr", architecture="cnn",
                                   architecture_specific='resnet31', pretraining_dataset="doctr", pretraining_mode="text_recognition", exclude_from_fe=True),

    "crnn_vgg16_bn": ModelProps(hugging=False, library="doctr", architecture="cnn",
                                architecture_specific='vgg16', pretraining_dataset="doctr", pretraining_mode="text_recognition", exclude_from_fe=True),
    
    "crnn_vgg16_bn_224": ModelProps(hugging=False, library="doctr", architecture="cnn",
                                     architecture_specific='vgg16', pretraining_dataset="doctr",pretraining_mode="text_recognition"),
    
    "crnn_vgg16_bn_224-inter": ModelProps(hugging=False, library="doctr", architecture="cnn",
                                          architecture_specific='vgg16', pretraining_dataset="doctr",pretraining_mode="text_recognition"),

    "vitstr_base": ModelProps(hugging=False, library="doctr", architecture="transformer",
                              architecture_specific='vit-b', pretraining_dataset="doctr", pretraining_mode="text_recognition", exclude_from_fe=True),

    "vitstr_base_224":ModelProps(hugging=False, library="doctr", architecture="transformer",
                              architecture_specific='vit-b', pretraining_dataset="doctr", pretraining_mode="text_recognition", exclude_from_fe=True),

    "vitstr_small": ModelProps(hugging=False, library="doctr", architecture="transformer",
                              architecture_specific='vit-s', pretraining_dataset="doctr", pretraining_mode="text_recognition", exclude_from_fe=True),

    "vitstr_small_224": ModelProps(hugging=False, library="doctr", architecture="transformer",
                              architecture_specific='vit-s', pretraining_dataset="doctr", pretraining_mode="text_recognition", exclude_from_fe=True),

    "trocr-large-handwritten": ModelProps(hugging=True, library="huggingface", architecture="transformer",
                                          architecture_specific="beit-l", pretraining_dataset="trocr+iam", pretraining_mode="text_recognition"),

    "trocr-large-handwritten-inter": ModelProps(hugging=True, library="huggingface", architecture="transformer",
                                                architecture_specific="beit-l", pretraining_dataset="trocr+iam", pretraining_mode="text_recognition"),
    
    "trocr-large-stage1": ModelProps(hugging=True, library="huggingface", architecture="transformer",
                                      architecture_specific="beit-l", pretraining_dataset="trocr", pretraining_mode="text_recognition"),
    
    "trocr-large-stage1-inter": ModelProps(hugging=True, library="huggingface", architecture="transformer",
                                            architecture_specific="beit-l", pretraining_dataset="trocr", pretraining_mode="text_recognition"),
    
    "trocr-base-handwritten": ModelProps(hugging=True, library="huggingface", architecture="transformer",
                                         architecture_specific="beit-b", pretraining_dataset="trocr+iam", pretraining_mode="text_recognition"),
    
    "trocr-base-stage1": ModelProps(hugging=True, library="huggingface", architecture="transformer",
                                     architecture_specific="beit-b", pretraining_dataset="trocr", pretraining_mode="text_recognition"),
    
    "trocr-small-handwritten": ModelProps(hugging=True, library="huggingface", architecture="transformer",
                                          architecture_specific="deit-s", pretraining_dataset="trocr+iam", pretraining_mode="text_recognition"),
    
    "trocr-small-stage1": ModelProps(hugging=True, library="huggingface", architecture="transformer",
                                      architecture_specific="deit-s", pretraining_dataset="trocr", pretraining_mode="text_recognition"),
    ### detection ###
    "dresnet50": ModelProps(hugging=False, library="doctr", architecture="cnn",
                            architecture_specific="resnet50", pretraining_dataset="doctr",pretraining_mode="text_detection"),

    "dresnet50-inter": ModelProps(hugging=False, library="doctr", architecture="cnn",
                            architecture_specific="resnet50", pretraining_dataset="doctr",pretraining_mode="text_detection"),

    "db_mobilenet": ModelProps(hugging=False, library="doctr", architecture="cnn",
                            architecture_specific="mobilenet_v3_l", pretraining_dataset="doctr",pretraining_mode="text_detection"),

    "linknet_resnet50_224": ModelProps(hugging=False, library="doctr", architecture="cnn",
                            architecture_specific="resnet50", pretraining_dataset="doctr",pretraining_mode="text_detection",exclude_from_fe=True),

    "linknet_resnet50": ModelProps(hugging=False, library="doctr", architecture="cnn",
                            architecture_specific="resnet50", pretraining_dataset="doctr",pretraining_mode="text_detection"),

    "linknet_resnet18": ModelProps(hugging=False, library="doctr", architecture="cnn",
                            architecture_specific="resnet18", pretraining_dataset="doctr",pretraining_mode="text_detection"),

    ### cnn models ###
    "resnet50": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                           architecture_specific="resnet50",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "resnet18": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                           architecture_specific="resnet18",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "googlenet": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                            architecture_specific="googlenet",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "vgg16": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                        architecture_specific="vgg16",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "alexnet": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                          architecture_specific="alexnet",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    ###
    "alexnet_gap": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                          architecture_specific="alexnet",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "densenet161": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                              architecture_specific="densenet161",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
                              
    "densenet121": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                              architecture_specific="densenet121",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "densenet201": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                              architecture_specific="densenet201",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "vgg11": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                        architecture_specific="vgg11",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "vgg16_512": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                            architecture_specific="vgg16_512",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "maxvit": ModelProps(hugging=False, library="torchvision",architecture="hybrid",
                         architecture_specific="maxvit",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "inception_net": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                                 architecture_specific="inception_net",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "efficientnet_v2_l": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                                      architecture_specific="efficientnet_v2_l",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "efficientnet_v2_s": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                                      architecture_specific="efficientnet_v2_s",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "convnext_large": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                                  architecture_specific="convnext_large",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "convnext_large-inter": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                                        architecture_specific="convnext_large-inter",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "convnext_base": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                                 architecture_specific="convnext_base",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "convnext_small": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                                  architecture_specific="convnext_small",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "mobilenet_v3_small": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                                      architecture_specific="mobilenet_v3_small",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

    "mobilenet_v3_large": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                                      architecture_specific="mobilenet_v3_large",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "resnet34_layer3": ModelProps(hugging=False, library="torchvision",architecture="cnn",
                                  architecture_specific="resnet34",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "resnet34_layer2": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                                  architecture_specific="resnet34",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "resnet34_layer1": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                                  architecture_specific="resnet34",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "resnet101": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                            architecture_specific="resnet101",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "resnet34": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                            architecture_specific="resnet34",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "regnet_x_32": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                              architecture_specific="regnet_x_32",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "regnet_x_3_2": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                               architecture_specific="regnet_x_3_2",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "regnet_y_128": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                               architecture_specific="regnet_y_128",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "regnet_y_32": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                              architecture_specific="regnet_y_32",pretraining_dataset="imagenet1k",pretraining_mode="classification"),
    
    "regnet_y_3_2": ModelProps(hugging=False, library="torchvision",architecture="cnn", 
                               architecture_specific="regnet_y_3_2",pretraining_dataset="imagenet1k",pretraining_mode="classification"),

}   

@lru_cache(maxsize=None)
def get_props(model_name: str) -> ModelProps:
    # Normalize keys if you have variants/aliases
    #key = model_name.strip().lower()
    try:
        return _MODELS[model_name]
    except KeyError as e:
        raise KeyError(f"Unknown model: {model_name}") from e

def list_models() -> list[str]:
    return list(_MODELS.keys())


