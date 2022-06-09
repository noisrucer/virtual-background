'''
Test on your local environment with camera support!!!
'''

import torch
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from model import BiSeNet
from utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint_path = 'model_weights/checkpoint-epoch9.pth'
background_path = 'data/alphaca.jpeg'

def prepare_model(checkpoint_path):
    model = BiSeNet(num_classes=1, training=False)
    model = model.to(device)
    load_model(model, checkpoint_path)
    model.eval()
    return model

def resize_img(img, h=512, w=512):
    img = A.Resize(512, 512)(image=img)['image']
    return img

def inference_single_img(model, img_np_array):
    transform = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2()
                ])

    img = transform(image=img_np_array)['image'].to(device)
    img = img.unsqueeze(dim=0)
    mask = model(img).squeeze()
    mask = torch.sigmoid(mask).detach().cpu().numpy()
    mask = (mask > 0.5) * 1
    return mask

model = prepare_model(checkpoint_path)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(10, 100)

print("Frame width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    success, img  = cap.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    background = cv2.imread(background_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background = resize_img(background)

    mask = inference_single_img(model, img)
    mask = np.expand_dims(mask, axis=0)
    mask = np.vstack((mask, mask, mask)).transpose((1, 2, 0))
    img = resize_img(img)
    masked_img = np.where(mask == 0, background, img)
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)

    cv2.imshow("webcam", masked_img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


