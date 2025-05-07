
from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms


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

def get_resnet_transforms(use_patches=False):
    """
    Returns the transformation pipeline for ResNet.
    """
    if use_patches:
        transform_resnet18 = transforms.Compose([
            transforms.Resize(256,interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            #transforms.Pad((0, 0, 0, 0), fill=0, padding_mode="constant"),  # Optional: Add padding if needed
            #weights["efficient"].transforms()
        ])
    else:
        transform_resnet18 = transforms.Compose([
            transforms.Resize(500,interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            #transforms.Pad((0, 0, 0, 0), fill=0, padding_mode="constant"),  # Optional: Add padding if needed
            #weights["efficient"].transforms()
        ])
    
    return transform_resnet18


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

def get_transform(name='resnet18',use_patches=True):
    if name=='resnet18':
        return get_resnet_transforms(use_patches=use_patches)
    elif name=='efficientnet':  
        return get_efficient_transforms()
    elif name=='handwriting':
        return get_handwriting_transforms()
    elif name=='smallcnn':
        return get_small_cnn_transforms()
    elif name=='mnist':
        return get_mnist_transforms()
    elif name=='Deit':
        return get_resnet_transforms(use_patches=use_patches)  # Assuming Deit uses the same transform as ResNet without patches
    else:
        raise ValueError(f"Unknown model name: {name}. Please provide a valid model name.")
