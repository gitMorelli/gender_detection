
from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms
from transformers import TrOCRProcessor, ViTImageProcessor
from doctr.models.preprocessor import PreProcessor
from PIL import Image, ImageOps

# Step 1: Define your resize_with_padding function
def resize_with_padding(image, target_size=(32, 128)):
    width, height = image.size
    #print(image.size)
    top = height // 3
    bottom = 2 * height // 3
    image = image.crop((0, top, width, bottom))  # (left, top, right, bottom)
    #print(image.size)
    #image.thumbnail(target_size, Image.ANTIALIAS)
    image.thumbnail((target_size[1],target_size[0]), Image.Resampling.LANCZOS)
    #print(image.size)
    delta_w = target_size[1] - image.width
    delta_h = target_size[0] - image.height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(image, padding, fill=0)


def get_mnist_transforms():
    """
    Returns the transformation pipeline for MNIST dataset.
    """

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean & std of MNIST
    ])
    
    return mnist_transform

def get_efficient_transforms():
    """
    Returns the transformation pipeline for EfficientNet.
    """
    efficient_transform = transforms.Compose([
        transforms.Resize(500,interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        #transforms.Pad((0, 0, 0, 0), fill=0, padding_mode="constant"),  # Optional: Add padding if needed
        #weights["efficient"].transforms()
    ])
    
    return efficient_transform

def get_resnet_transforms(name, use_patches=False, **kwargs):
    """
    Returns the transformation pipeline for ResNet.
    """
    mode=kwargs.get('mode',)
    if kwargs.get('custom')==True:
        if name in ['resnet18','resnet50']:
            if use_patches:
                transform = transforms.Compose([
                    transforms.Resize(256,interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    #transforms.Pad((0, 0, 0, 0), fill=0, padding_mode="constant"),  # Optional: Add padding if needed
                    #weights["efficient"].transforms()
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(500,interpolation=InterpolationMode.BILINEAR),
                    #transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    #transforms.Pad((0, 0, 0, 0), fill=0, padding_mode="constant"),  # Optional: Add padding if needed
                    #weights["efficient"].transforms()
                ])
        else:
            raise ValueError(f"Model {name} is not supported for custom transforms.")
    else:
        if name in ['resnet18','resnet50']:
            if mode=='resize':
                transform = transforms.Compose([
                    transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR),
                    #transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            raise ValueError(f"Model {name} is not supported.")
    return transform

def get_trocr_transforms(name='trocr-small-stage1'):
    processor = TrOCRProcessor.from_pretrained(f'microsoft/{name}',use_fast=False)
    return processor

def get_vit_transforms(name='vit-base-patch16-224-in21k'):
    if name in ['vit-base-patch16-224-in21k']:
        processor = ViTImageProcessor.from_pretrained(f'google/{name}')
    else:
        raise ValueError(f"Model {name} is not supported.")
    return processor

def get_handwriting_transforms():
    """
    Returns the transformation pipeline for Handwriting CNN.
    """
    handwriting_transform = transforms.Compose([
        transforms.Resize(500,interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        #transforms.Pad((0, 0, 0, 0), fill=0, padding_mode="constant"),  # Optional: Add padding if needed
        #weights["efficient"].transforms()
    ])
    
    return handwriting_transform

def get_small_cnn_transforms():
    """
    Returns the transformation pipeline for Small CNN.
    """
    small_cnn_transform = transforms.Compose([
        transforms.Resize(500,interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        #transforms.Pad((0, 0, 0, 0), fill=0, padding_mode="constant"),  # Optional: Add padding if needed
        #weights["efficient"].transforms()
    ])
    
    return small_cnn_transform

def get_dresnet50_transforms(**kwargs):
    ''' doctr default preprocessor
    transform=PreProcessor(
        (1024, 1024),
        batch_size=1,
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    )'''
    if kwargs.get('custom')==True:
        print('no support for custom transforms')
    else:
        transform = transforms.Compose([
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.798, 0.785, 0.772], std=[0.264, 0.2749, 0.287]),
        ])
    return transform

def get_crnn_vgg16_bn_transforms(**kwargs):
    #Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))
    mode=kwargs.get('mode',)
    if kwargs.get('custom')==True:
        mode=kwargs.get('mode','padding')
        if mode=='padding':
            transform = transforms.Compose([
                transforms.Lambda(lambda img: resize_with_padding(img, target_size=(32, 128))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.694, 0.695, 0.693], std=[0.299, 0.296, 0.301]),
            ])
        elif mode == 'crop':
            transform = transforms.Compose([
                transforms.Resize(128, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((32, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.694, 0.695, 0.693], std=[0.299, 0.296, 0.301]),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize((32,128), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.694, 0.695, 0.693], std=[0.299, 0.296, 0.301]),
        ])
    return transform

def get_sar_resnet31_transforms(**kwargs):
    #Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))
    if kwargs.get('custom')==True:
        print('no support for custom transforms')
    else:
        transform = transforms.Compose([
            transforms.Resize((32,128), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.694, 0.695, 0.693], std=[0.299, 0.296, 0.301]),
        ])
    return transform

def get_vitstr_base_transforms(**kwargs):
    #Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))
    if kwargs.get('custom')==True:
        print('no support for custom transforms')
    else:
        transform = transforms.Compose([
            transforms.Resize((32,128), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.694, 0.695, 0.693], std=[0.299, 0.296, 0.301]),
        ])
    return transform

def get_layoutlmv3_base_transforms(**kwargs):
    from transformers import LayoutLMv3Processor
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    return processor

def get_transform(name='resnet18',use_patches=True, **kwargs):
    if name in ['resnet18','resnet50','resnet101','resnet152']:
        return get_resnet_transforms(name,use_patches=use_patches,**kwargs)
    elif name=='efficientnet':  
        return get_efficient_transforms()
    elif name=='handwriting':
        return get_handwriting_transforms()
    elif name=='smallcnn':
        return get_small_cnn_transforms()
    elif name=='mnist':
        return get_mnist_transforms()
    elif name in ['trocr-small-stage1','trocr-small-handwritten','trocr-base-handwritten']:
        return get_trocr_transforms(name)  # Assuming Deit uses the same transform as ResNet without patches
    elif name in ['vit-base-patch16-224-in21k']:
        return get_vit_transforms(name)  # Assuming Deit uses the same transform as ResNet without patches
    elif name=='dresnet50':
        return get_dresnet50_transforms(**kwargs)
    elif name=='crnn_vgg16_bn':
        return get_crnn_vgg16_bn_transforms(**kwargs)
    elif name=='sar_resnet31':
        return get_sar_resnet31_transforms(**kwargs)
    elif name=='vitstr_base':
        return get_vitstr_base_transforms(**kwargs)
    elif name=='layoutlmv3_base':
        return get_layoutlmv3_base_transforms(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}. Please provide a valid model name.")
