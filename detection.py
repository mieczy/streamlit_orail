import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
import pandas as pd

from functools import lru_cache

from PIL import Image

from typing import Final

import logging


class OrailModel:
    DIR: Final = './ml/'

    def __init__(self, name):
        self.model = densenet121(num_classes=5) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.DIR+name, map_location=torch.device('cpu')))
        self.model.eval()
        self.model.to(self.device)
        self.model.eval()


    def convert_image(self, file):
        with (Image.open(file) as image):
            
            image_rgb = image.convert('RGB') # RGB 변환

            resizer = transforms.Resize((32, 32))
            image_resized = resizer(image_rgb)

            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.9937, 0.9961, 0.9980], std=[0.0314, 0.0191, 0.0105]),
                ])

            input_image = transform(image_resized).unsqueeze(0).to(self.device)

        return input_image
    

    def inference_single_image(self, input_image):
        with torch.no_grad():
            input_image = input_image.to(self.device)
            y_pred = self.model(input_image)
            _, predicted = torch.max(y_pred.data, 1)
            probabilities = torch.softmax(y_pred, dim=1)

        predicted_label = predicted.item()
        label_probabilities = [prob.item()*100 for i, prob in enumerate(probabilities[0])]
        
        return predicted_label, label_probabilities
    
        
    def detect(self, image_file):
        try:
            input_image = self.convert_image(image_file)
            predicted_label, label_probabilities = self.inference_single_image(input_image)
            
            p = label_probabilities[predicted_label]
            if p <= 75:
                is_odd = 'Yes (Score1)'
            elif p <= 85:
                is_odd = 'Yes (Score2)'
            else:
                is_odd = 'No'
            
            return predicted_label, label_probabilities, is_odd
        
        except Exception  as e:
            logging.error(e)
            raise Exception()
        
@lru_cache
def get_model(name: str):
    return OrailModel(name)