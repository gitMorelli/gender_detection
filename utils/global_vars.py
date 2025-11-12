
# models.py
from dataclasses import dataclass
from functools import lru_cache

@dataclass(frozen=True)
class ModelProps:
    hugging: bool #True, False
    library: str #torchvision, doctr, huggingface
    architecture: str #transformer, cnn, hybrid
    exclude_from_fe: bool = False
    # If you have optional/unknown extras:
    # extras: dict[str, object] = field(default_factory=dict)

# Single shared mapping (module-level constant)
_MODELS: dict[str, ModelProps] = {
    ### transformer models ###
    "swin_b":   ModelProps(hugging=False, library="torchvision", architecture="transformer"),
    "swin_s":   ModelProps(hugging=False, library="torchvision", architecture="transformer"),
    "DeiT-Small":  ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "DeiT-Small-Dist": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "DeiT-Base":   ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "BEiT-Base":   ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "BEiT-Large":  ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "BEiT-Large-inter":  ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "vit-base-patch16-224": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "vit-base-patch32-224-in21k": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "vit-large-patch16-224-in21k": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "vit-huge-patch14-224-in21k": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "vit-base-patch16-224-in21k": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "DeiT-Tiny":  ModelProps(hugging=True, library="huggingface", architecture="transformer"), 
    "clip-vit-base-patch16": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "clip-vit-base-patch32": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "clip-vit-large-patch14-un": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "clip-vit-large-patch14": ModelProps(hugging=True, library="huggingface", architecture="transformer",exclude_from_fe=True),
    "clip-vit-large-patch14-inter": ModelProps(hugging=True, library="huggingface", architecture="transformer"),

    ### ocr models ###
    ### recognition###
    "crnn_mobilenet": ModelProps(hugging=False, library="doctr", architecture="cnn",exclude_from_fe=True),
    "crnn_mobilenet_224": ModelProps(hugging=False, library="doctr", architecture="cnn"),
    "crnn_mobilenet_224-inter": ModelProps(hugging=False, library="doctr", architecture="cnn"),
    "sar_resnet31": ModelProps(hugging=False, library="doctr", architecture="cnn",exclude_from_fe=True),
    "sar_resnet31_224": ModelProps(hugging=False, library="doctr", architecture="cnn"),
    "crnn_vgg16_bn": ModelProps(hugging=False, library="doctr", architecture="cnn",exclude_from_fe=True),
    "crnn_vgg16_bn_224": ModelProps(hugging=False, library="doctr", architecture="cnn"),
    "vitstr_base": ModelProps(hugging=False, library="doctr", architecture="transformer",exclude_from_fe=True),
    "vitstr_base_224": ModelProps(hugging=False, library="doctr", architecture="transformer"),
    "vitstr_small": ModelProps(hugging=False, library="doctr", architecture="transformer",exclude_from_fe=True),
    "vitstr_small_224": ModelProps(hugging=False, library="doctr", architecture="transformer"),
    "trocr-large-handwritten": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "trocr-large-handwritten-inter": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "trocr-large-stage1": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "trocr-large-stage1-inter": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "trocr-base-handwritten": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "trocr-base-stage1": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "trocr-small-handwritten": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    "trocr-small-stage1": ModelProps(hugging=True, library="huggingface", architecture="transformer"),
    ### detection ###
    "dresnet50": ModelProps(hugging=False, library="doctr", architecture="cnn"),
    "dresnet50-inter": ModelProps(hugging=False, library="doctr", architecture="cnn"),
    "db_mobilenet": ModelProps(hugging=False, library="doctr", architecture="cnn"),
    "linknet_resnet50_224": ModelProps(hugging=False, library="doctr", architecture="cnn",exclude_from_fe=True),
    "linknet_resnet50": ModelProps(hugging=False, library="doctr", architecture="cnn"),
    "linknet_resnet18": ModelProps(hugging=False, library="doctr", architecture="cnn"),

    ### cnn models ###
    "resnet50": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "resnet18": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "googlenet": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "vgg16": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "alexnet": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    ###
    "alexnet_gap": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "densenet161": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "densenet121": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "densenet201": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "vgg11": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "vgg16_512": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "maxvit": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "inception_net": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "efficientnet_v2_l": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "efficientnet_v2_s": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "convnext_large": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "convnext_large-inter": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "convnext_base": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "convnext_small": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "mobilenet_v3_small": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "mobilenet_v3_large": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "resnet34_layer3": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "resnet34_layer2": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "resnet34_layer1": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "resnet101": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "resnet34": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "regnet_x_32": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "regnet_x_3_2": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "regnet_y_128": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "regnet_y_32": ModelProps(hugging=False, library="torchvision",architecture="cnn"),
    "regnet_y_3_2": ModelProps(hugging=False, library="torchvision",architecture="cnn"),

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


