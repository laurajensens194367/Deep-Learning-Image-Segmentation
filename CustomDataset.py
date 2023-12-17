from torch.utils.data import Dataset
from PIL import Image  # Assuming you have PIL installed
from torchvision import transforms
import torch
 
class CustomDataset(Dataset):
    def __init__(self, data, transform=None, target_transform = None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_image = self.data[idx][:, :, :3]  # Assuming the first three layers are RGB
        target = self.data[idx][:, :, 3]  # Assuming the fourth layer is the target
        target = target*0.1  # to devide target values with 10
        target[target == 9] = 0 # converting the outer parts of the cars to have value 0
        rgb_image = rgb_image / 255.0
        
        
        rgb_image = torch.from_numpy(rgb_image.astype('float32'))
        target = torch.LongTensor(target)
        
        rgb_image = rgb_image.permute(2, 0, 1)
        #sample = {
        #    'image': rgb_image,
        #    'target': target
        #}

        if self.transform:
            rgb_image = self.transform(rgb_image)
            
        if self.target_transform:
            target = self.transform(target)
            
        
        return rgb_image,target