import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class QwenImagePreprocessing:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_pixels": ("INT", {
                    "default": 1024, 
                    "min": 0, 
                    "max": 8192, 
                    "step": 64, 
                    "display": "number",
                    "tooltip": "若图像短边低于此像素值，将触发放大机制。设为0则不强制放大。"
                }),
                "constraint_mode": ([
                    "Strict Qwen (28x)", 
                    "Balanced (56x) - Rec. for SD1.5/XL", 
                    "Safe (112x) - Rec. for Video/Flux",
                    "Extreme (224x)"
                ], {
                    "default": "Balanced (56x) - Rec. for SD1.5/XL",
                    "tooltip": "选择尺寸对齐策略：\n- 28x: 仅满足Qwen。\n- 56x: 同时满足Qwen和8的倍数（推荐）。"
                }),
                # 在这里添加了 lanczos
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bicubic"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "process_image"
    CATEGORY = "Qwen Image Edit"

    def process_image(self, image, min_pixels, constraint_mode, upscale_method):
        # image shape is [Batch, Height, Width, Channels]
        b, h, w, c = image.shape
        
        # 1. 确定对齐的倍数 (LCM 逻辑)
        if "Strict Qwen" in constraint_mode:
            multiple = 28
        elif "Balanced" in constraint_mode:
            multiple = 56
        elif "Safe" in constraint_mode:
            multiple = 112
        else:
            multiple = 224
            
        print(f"Qwen Preprocessor: Target multiple set to {multiple}")

        # 2. 计算缩放比例
        scale_factor = 1.0
        short_side = min(h, w)
        
        if min_pixels > 0 and short_side < min_pixels:
            scale_factor = min_pixels / short_side
        
        target_h = h * scale_factor
        target_w = w * scale_factor
        
        # 3. 计算对齐后的尺寸
        new_h = int(round(target_h / multiple) * multiple)
        new_w = int(round(target_w / multiple) * multiple)
        
        new_h = max(new_h, multiple)
        new_w = max(new_w, multiple)
        
        if new_h == h and new_w == w:
            return (image, w, h)
            
        print(f"Qwen Preprocessor: Resizing from {w}x{h} to {new_w}x{new_h} using {upscale_method}")

        # 4. 执行 Resize
        
        # --- 分支 A: Lanczos 算法 (使用 PIL 实现) ---
        if upscale_method == "lanczos":
            # ComfyUI 的 image 是 tensor float32 [0,1]，需要转换为 uint8 [0,255] 才能给 PIL 处理
            # 先转到 CPU 并转为 numpy
            img_np = image.cpu().numpy()
            
            output_list = []
            
            # 遍历 Batch 中的每一张图
            for i in range(b):
                # 转换: (H, W, C) float -> uint8
                single_img_np = (img_np[i] * 255.0).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(single_img_np)
                
                # 使用 PIL 的 Lanczos 滤镜
                pil_resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)
                
                # 转换回: (H, W, C) uint8 -> float32 tensor
                resized_np = np.array(pil_resized).astype(np.float32) / 255.0
                output_list.append(torch.from_numpy(resized_np))
            
            # 重新堆叠为 Batch Tensor: (B, H, W, C)
            final_image = torch.stack(output_list)
            
        # --- 分支 B: PyTorch 原生算法 (GPU加速) ---
        else:
            # 调整维度为 (B, C, H, W)
            img_permuted = image.permute(0, 3, 1, 2)
            
            if upscale_method == "nearest-exact":
                img_resized = F.interpolate(img_permuted, size=(new_h, new_w), mode='nearest')
            else:
                img_resized = F.interpolate(img_permuted, size=(new_h, new_w), mode=upscale_method, align_corners=False)
                
            # 转回 (B, H, W, C)
            final_image = img_resized.permute(0, 2, 3, 1)

        return (final_image, new_w, new_h)

NODE_CLASS_MAPPINGS = {
    "QwenImagePreprocessing": QwenImagePreprocessing
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImagePreprocessing": "Qwen Image Resize & Pad"
}