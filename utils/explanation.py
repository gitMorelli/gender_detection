##https://github.com/jacobgil/pytorch-grad-cam/tree/master
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
#from pytorch_grad_cam.utils.vit import reshape_transform
import torch
import numpy as np
from PIL import Image
import cv2
import pandas as pd

#cam explanations
def load_images(df,transform,huggingface):
    """
    Loads an image from the specified path and converts it to a tensor.
    """
    for idx, row in df.iterrows():
        image = Image.open(row['file_name']).convert('RGB')
        x1 = row['x']
        y1 = row['y']
        x2 = row['x2']
        y2 = row['y2']
        patch = image.crop((x1, y1, x2, y2))
        if huggingface:
            # the transform is actually an huggingface processor in this case
            inputs = transform(images=patch, return_tensors="pt")
            # Remove batch dimension from inputs
            patch = inputs['pixel_values'].squeeze()
        else:
            patch = transform(patch)
        transformed_image=patch.unsqueeze(0)
        if 'transformed_images' not in locals():
            transformed_images = transformed_image
        else:
            transformed_images = torch.cat((transformed_images, transformed_image), dim=0)
    return transformed_images
def select_layer(model,model_name,mode='last'):
    """
    Selects a specific layer from the model by its name.
    """
    if model_name == 'clip-vit-large-patch14':
        # For CLIP, you might want to select the last layer of the transformer
        return [model.vision_model.model.vision_model.encoder.layers[-1].layer_norm1]
    elif model_name == 'resnet50':
        # For ResNet, you might want to select the last convolutional layer
        if mode == 'last':
            return [model.vision_model.layer4[-1]]
        elif mode == 'first':
            return [model.vision_model.layer1[-1]]
    return   
def get_reshape_transform(model_name):
    """
    Returns a reshape transform function based on the model name.
    """
    if model_name == 'clip-vit-large-patch14':
        return reshape_transform  # Use the custom reshape_transform defined above
    elif model_name == 'resnet50':
        return None
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
def gradcam_on_df(df,model,transform,huggingface,device,model_name='clip-vit-large-patch14', mode='last'):
    model = model.to(device)
    # Get a batch of data from train_dataloader
    grouped = df.groupby('page', sort=False)
    #print(model)
    target_layers = select_layer(model,model_name,mode) # Usually the last conv layer
    model.eval()
    page_groups_visualizations = []
    #print(target_layers)
    for page, group in grouped:
        n_patches=len(group['grouped_true'].values)
        print(page,n_patches)
        labels = torch.tensor(group['y_pred'].values)  # Extract labels and convert to tensor
        #labels = torch.ones(n_patches, dtype=torch.long, device=device)
        #labels = torch.zeros(n_patches, dtype=torch.long, device=device)
        #predictions = torch.tensor(group['y_pred'].values)   # Extract predictions and convert to tensor
        # The batch correct shape is (batch_size, 3, H, W)

        reshape_transform = get_reshape_transform(model_name) 
        
        input_tensor = load_images(group, transform=transform, huggingface=huggingface)
        input_tensor.requires_grad = True
        input_tensor = input_tensor.to(device)
        
        visualizations = gradcam_on_batch(model, input_tensor, labels, target_layers, reshape_transform=reshape_transform)
        page_groups_visualizations.append(visualizations)
    return page_groups_visualizations
def gradcam_on_batch(model, input_tensor, labels, target_layers, reshape_transform=None):
    # Specify the target layers and targets
    targets = [ClassifierOutputTarget(label.item()) for label in labels]  # Create targets for each image in the batch

    # Construct the CAM object and apply it to the batch
    if reshape_transform:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=target_layers)
    with cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        
        # Visualize each image in the batch
        visualizations = []
        for i in range(len(labels)):
            #rgb_image = torch.clamp(input_tensor[i].permute(1, 2, 0), 0, 1).cpu().detach().numpy().astype(np.float32)
            image = input_tensor[i].permute(1, 2, 0)  # CHW -> HWC
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                rgb_image = ((image - min_val) / (max_val - min_val)).cpu().detach().numpy().astype(np.float32)
            else:
                rgb_image = torch.zeros_like(image).cpu().detach().numpy().astype(np.float32)  # fallback if image is constant

            grayscale_cam = grayscale_cams[i]
            visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            visualizations.append(visualization)
    return visualizations
def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
def check_gradients(model, input_tensor, target_tensor, loss_fn):
    """
    Checks if gradients are flowing through the model.
    Args:
        model: The PyTorch model.
        input_tensor: Input tensor with requires_grad=True.
        target_tensor: Target tensor for loss computation.
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss()).
    Returns:
        grad_status: Dict mapping parameter names to whether gradients are non-zero.
    """
    model.zero_grad()
    output = model(input_tensor)
    loss = loss_fn(output, target_tensor)
    loss.backward()
    grad_status = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_status[name] = param.grad.abs().sum().item() > 0
        else:
            grad_status[name] = False
    return grad_status

# attention maps
def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9, model_name='clip-vit-large-patch14'):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        '''for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)'''
        if model_name == 'clip-vit-large-patch14':
            for layer in model.vision_model.encoder.layers:
                layer.self_attn.register_forward_hook(self.save_attention_hook)
        else:
            raise ValueError(f"Model {model_name} is not supported for VITAttentionRollout. Only 'clip-vit-large-patch14' is supported.")

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())
    
    def save_attention_hook(self,module, input, output):
        attn_weights = output[1]
        if attn_weights is not None:
            self.attentions.append(attn_weights.detach().cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)

def attention_on_df(df,model,transform,huggingface,device,model_name='clip-vit-large-patch14'):
    # Get a batch of data from train_dataloader
    grouped = df.groupby('page', sort=False)
    #print(model)
    model.eval()
    model=model.to(device)
    rollout = VITAttentionRollout(model, discard_ratio=0.9, head_fusion='max',attention_layer_name='attn_drop')
    page_groups_visualizations = []
    for page, group in grouped:
        n_patches=len(group['grouped_true'].values)
        print(page,n_patches)
        labels = torch.tensor(group['y_pred'].values)  # Extract labels and convert to tensor
        input_tensor = load_images(group, transform=transform, huggingface=huggingface)
        input_tensor.requires_grad = True
        input_tensor = input_tensor.to(device)
        visualizations = attention_on_batch(rollout, input_tensor, labels)
        page_groups_visualizations.append(visualizations)
    return page_groups_visualizations
def attention_on_batch(rollout, input_tensor, labels):
    visualizations = []
    for i in range(len(labels)):
        #print(input_tensor[i].shape)
        mask = rollout(input_tensor[i].unsqueeze(0))#, category_index=0)
        visualization = show_mask_on_image(input_tensor[i].unsqueeze(0), mask)
        visualizations.append(visualization)
    return visualizations

def show_mask_on_image(input_tensor, mask):
    np_img = input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    mask_r=cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    #img = np.float32(np_img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_r), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(np_img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)