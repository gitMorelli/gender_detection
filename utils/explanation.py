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
def rollout(attentions, discard_ratio, head_fusion, protect_cls=True,
            norm="percentile", pct_lo=5.0, pct_hi=95.0, squeeze=True):
    """
    Attention rollout over a list of per-layer attention probability tensors.

    attentions : list of (B, heads, N, N), in forward execution order
    norm       : "percentile" | "max" | "none"
    pct_lo/hi  : percentile bounds for norm="percentile"
    squeeze    : drop the batch dim when B == 1, so downstream cv2.resize
                 receives a genuine 2-D array
    """
    N = attentions[0].size(-1)
    B = attentions[0].size(0)
    result = torch.eye(N).unsqueeze(0).expand(B, -1, -1).clone()
    I = torch.eye(N).unsqueeze(0)

    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                fused = attention.mean(dim=1)
            elif head_fusion == "max":
                fused = attention.max(dim=1)[0]
            elif head_fusion == "min":
                fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Unsupported head_fusion: {head_fusion}")

            fused = fused.clone()
            flat = fused.view(B, -1)
            k = int(flat.size(-1) * discard_ratio)
            if k > 0:
                _, idx = flat.topk(k, dim=-1, largest=False)
                if protect_cls:
                    keep = (idx % N != 0) & (idx // N != 0)
                else:
                    keep = torch.ones_like(idx, dtype=torch.bool)
                for b in range(B):
                    flat[b, idx[b][keep[b]]] = 0

            a = (fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)

    mask = result[:, 0, 1:]                        # (B, N-1)

    width = int(round(mask.size(-1) ** 0.5))
    if width * width != mask.size(-1):
        raise ValueError(f"{mask.size(-1)} patch tokens is not a square grid")
    mask = mask.reshape(B, width, width)

    if norm == "percentile":
        flat_m = mask.reshape(B, -1)
        lo = torch.quantile(flat_m, pct_lo / 100.0, dim=1).view(B, 1, 1)
        hi = torch.quantile(flat_m, pct_hi / 100.0, dim=1).view(B, 1, 1)
        mask = (mask - lo) / (hi - lo + 1e-8)
        mask = mask.clamp(0.0, 1.0)
    elif norm == "max":
        mask = mask / (mask.amax(dim=(1, 2), keepdim=True) + 1e-8)
    elif norm != "none":
        raise ValueError(f"Unsupported norm: {norm}")

    mask = mask.numpy().astype(np.float32)
    if squeeze and mask.shape[0] == 1:
        mask = mask[0]                             # (width, width)
    return mask



class CheferRelevance:
    def __init__(self, model, start_layer=0, end_layer=None, model_name=None):
        self.model = model
        self.attns, self.grads = [], []
        self.handles = []

        layers = model.vision_model.vision.encoder.layers
        if model_name == 'clip-vit-large-patch14':
            end_layer=24
        elif model_name == 'clip-vit-large-patch14-inter':
            end_layer=12
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.n_expected = end_layer - start_layer

        for layer in layers[start_layer:end_layer]:
            self.handles.append(
                layer.self_attn.register_forward_hook(self._fwd)
            )
        model.vision_model.vision.config.output_attentions = True

    def _fwd(self, module, inp, out):
        attn = out[1]
        if attn is None:
            raise RuntimeError("attn is None — need attn_implementation='eager'")
        attn.retain_grad()              # non-leaf tensor: keep .grad after backward
        self.attns.append(attn)         # NO .detach(), NO .cpu() — must stay in graph

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, pixel_values, target_fn):
        """
        target_fn: callable taking the model output, returning a scalar per batch
                   element to explain (a class logit, or an image-text similarity).
        """
        self.attns = []
        self.model.zero_grad()

        out = self.model(pixel_values)
        scalar = target_fn(out)
        scalar.backward(retain_graph=False)

        assert len(self.attns) == self.n_expected, \
            f"got {len(self.attns)}, expected {self.n_expected}"

        B, _, N, _ = self.attns[0].shape
        device = self.attns[0].device
        R = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        for attn in self.attns:
            grad = attn.grad
            if grad is None:
                raise RuntimeError("no grad on attention — target not differentiable "
                                   "w.r.t. the vision encoder?")
            cam = (grad * attn).clamp(min=0).mean(dim=1)     # (B, N, N)
            R = R + torch.bmm(cam, R)

        mask = R[:, 0, 1:]                                   # CLS row
        width = int(round(mask.size(-1) ** 0.5))
        mask = mask.reshape(B, width, width)

        flat = mask.reshape(B, -1)
        lo = torch.quantile(flat, 0.05, dim=1).view(B, 1, 1)
        hi = torch.quantile(flat, 0.95, dim=1).view(B, 1, 1)
        mask = ((mask - lo) / (hi - lo + 1e-8)).clamp(0, 1)

        mask = mask.detach().cpu().numpy().astype(np.float32)
        return mask[0] if mask.shape[0] == 1 else mask

class VITAttentionRollout:
    def __init__(self, model, head_fusion="mean", discard_ratio=0.9,
                 start_layer=0, end_layer=None, model_name=None):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.model_name = model_name
        self.attentions = []
        self.handles = []

        layers = model.vision_model.encoder.layers
        if model_name == 'clip-vit-large-patch14':
            end_layer=24
        elif model_name == 'clip-vit-large-patch14-inter':
            end_layer=11
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        for layer in layers[start_layer:end_layer]:
            h = layer.self_attn.register_forward_hook(self.save_attention_hook)
            self.handles.append(h)

        model.vision_model.config.output_attentions = True

    def save_attention_hook(self, module, inp, out):
        attn = out[1]
        if attn is None:
            raise RuntimeError("attn weights are None — need eager attention")
        self.attentions.append(attn.detach().cpu())

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            self.model(input_tensor)
        
        return rollout(self.attentions, self.discard_ratio, self.head_fusion)


def attention_on_df(df, model, transform, huggingface, device,
                    method='rollout', model_name='clip-vit-large-patch14', **kwargs):
    grouped = df.groupby('page', sort=False)
    model.eval().to(device)
    page_groups_visualizations = []

    if method == 'chefer':
        ctx = CheferRelevance(model, start_layer=0, end_layer=12,model_name=model_name)
    elif method == 'rollout':
        ctx = VITAttentionRollout(
            model,
            discard_ratio=kwargs.get('discard_ratio', 0.9),
            head_fusion=kwargs.get('head_fusion', 'max'),
            end_layer=12, model_name=model_name)
    else:
        raise ValueError(f"Unknown method: {method}")

    with ctx as explainer:
        for page, group in grouped:
            labels = torch.tensor(group['y_pred'].values)
            input_tensor = load_images(group, transform=transform,
                                       huggingface=huggingface).to(device)
            page_groups_visualizations.append(
                attention_on_batch(explainer, input_tensor, labels, method=method)
            )
    return page_groups_visualizations


def attention_on_batch(explainer, input_tensor, labels, method='chefer'):
    visualizations = []
    for i in range(len(labels)):
        x = input_tensor[i].unsqueeze(0)
        if method == 'chefer':
            cls_idx = int(labels[i])
            mask = explainer(x, target_fn=lambda logits: logits[0, cls_idx])
            '''mask_0 = explainer(x, target_fn=lambda logits: logits[0, 0])  # for class 0
            mask_1 = explainer(x, target_fn=lambda logits: logits[0, 1])  # for class 1
            mask_diff = mask_1 - mask_0  # difference between class 1 and class 0
            mask = mask_diff  # use the difference as the final mask '''
        else:
            mask = explainer(x)
        visualizations.append(mask)
        '''probe = np.zeros((16, 16), dtype=np.float32)
                probe[:4, :] = 1.0
                visualizations.append(probe)'''
    return visualizations
