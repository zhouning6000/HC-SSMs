import torch

# 加载 checkpoint
checkpoint = torch.load('/root/autodl-tmp/EarthFarseer-main/best_model_weights_moving.pth', map_location='cpu')


# 如果保存的是整个模型对象，并且有 __class__.__name__
if 'Earthfarseer_model' in checkpoint:
    checkpoint['HC_model'] = checkpoint.pop('Earthfarseer_model')

torch.save(checkpoint, r"/root/autodl-tmp/EarthFarseer-main/best_model_weights_moving_HC.pth")