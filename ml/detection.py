import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121

from os.path import isfile
from PIL import Image

from typing import Final


class OrailModel:
    DIR: Final = './ml/'
    CLASS_NAMES: Final = ['NORMAL', 'AVG', 'MULTITREND', 'HUNTING', 'DRIFT']

    def __init__(self):
        self.model = densenet121(num_classes=5) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.DIR+'Densenet121.pth', map_location=torch.device('cpu')))
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
        label_probabilities = {self.CLASS_NAMES[i]: prob.item() for i, prob in enumerate(probabilities[0])}

        return predicted_label, label_probabilities
    
    def get_result(self, predicted_label, label_probabilities):
        # ft. optional label_probabilities 
        # ft. score/confident level calculate
        if predicted_label == 0:
            result = 'Normal Type'
            odd_image = 'No'
        else :
            result = 'Abnormal Type '  + str(predicted_label) + ': ' + self.CLASS_NAMES[predicted_label]
            odd_image = 'Yes'

        return result, odd_image
    
        
    def detect(self, image_file):
        try:
            input_image = self.convert_image(image_file)
            predicted_label, label_probabilities = self.inference_single_image(input_image)
        
            return self.get_result(predicted_label, label_probabilities)
        except Exception  as e:
            raise Exception()
