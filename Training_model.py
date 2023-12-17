
import os
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import pickle
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from CustomDataset import *
import fnmatch
import random

############### Load data ##################


# Assuming 'data' contains your numpy arrays
#data_folder = paths_arrays = "c:/Users/Laura/Dropbox/Skole/DTU/9.semester/Deep Learning/Exercises/Project/carseg_data (3)/carseg_data/arrays/"

data_folder = "/zhome/50/8/147315/DL_Project/arrays"
paths = [os.path.join(data_folder, name) for name in os.listdir(data_folder)]
paths_photo = [os.path.join(data_folder, name) for name in os.listdir(data_folder) if fnmatch.fnmatch(name, 'photo*.npy')]

random.seed(123)

get_test_paths = random.sample(paths_photo, 25)

for i in range(len(get_test_paths)):
    if get_test_paths[i] in paths:
        paths.remove(get_test_paths[i])
        
        
data = [np.load(file_path) for file_path in paths]
test_data = [np.load(file_path) for file_path in get_test_paths]


# Split data into training and testing sets
train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)


transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1)
    ]
)

#custom_transform = ToTensorAndTargetTransform()

# Create instances of CustomDataset for training and testing
train_dataset = CustomDataset(train_data, transform=transform, target_transform = None)
validation_dataset = CustomDataset(validation_data, transform=transform, target_transform = None)
test_dataset = CustomDataset(test_data, transform=transform, target_transform = None)
   
    
#################### Dataloader ############################


batch_size = 3

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Using cuda')
else:
    device = torch.device('cpu')
    print('Using cpu')
    
    
##################### Import the model #########################    
#from models.DeepLab_2.deeplab import *
from models.UNET.unet_model import *


num_epochs = 100  
#modelname = "DeepLab_%sE"%(num_epochs)
modelname = "Unet_%sE"%(num_epochs)
    
n_channels = 1
n_classes = 9 # We now only have 9 classes, because class 9 is changed to 0    
net = UNet(n_channels, n_classes)

#net = DeepLab(backbone='resnet', output_stride=16, num_classes=9,sync_bn=True, freeze_bn=False)

net.to(device)

    
######################  Accuracy and Loss function ####################

def acc_score(target, output):
    target = target.cpu().numpy()
    #preds = torch.argmax(output, dim=1).cpu().data.numpy()
    preds = torch.max(output, 1)[1] # find max af hver pixel
    preds = preds.cpu().numpy()
    
    # Calculate accuracy for the background (class 0)
    acc_background = f1_score(target.flatten(), preds.flatten(), labels=[0], average='micro')

    # Calculate accuracy for the classes (excluding background)
    acc_classes = f1_score(target.flatten(), preds.flatten(), labels=list(range(1, 9)), average='micro')

    # Weighted sum of accuracies
    weighted_acc = (acc_background / 9) + (acc_classes * 8 / 9)

    return weighted_acc

lr = 0.001


# Class weights for CrossEntropyLoss
class_labels = np.concatenate([target.flatten() for image,target in train_dataset])
class_weights = compute_class_weight(class_weight ='balanced', classes = np.unique(class_labels), y=class_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer function
optimizer = torch.optim.Adam(net.parameters(), lr=lr)



####################### Training Loop ###################################

validation_every_steps = 500

step = 0
net.train()

train_accuracies = []
valid_accuracies = []
        
for epoch in range(num_epochs):
    
    train_accuracies_batches = []
    
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass, compute gradients, perform one training step.
        optimizer.zero_grad()
        output = net(inputs)
        
        loss = loss_function(output,targets)
        
        loss.backward()
        optimizer.step()
    
        
        # Increment step counter
        step += 1
        
        
        # Compute accuracy.
        #predictions = output.max(1)[1]
        
        
        train_accuracies_batches.append(acc_score(targets, output))
        
        if step % validation_every_steps == 0:
            
            # Append average training accuracy to list.
            train_accuracies.append(np.mean(train_accuracies_batches))
            
            train_accuracies_batches = []
        
        
            # Compute accuracies on validation set.
            valid_accuracies_batches = []
            
            with torch.no_grad():
                net.eval()
                for inputs, targets in validation_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = net(inputs)
                    loss = loss_function(output, targets)

                    #predictions = output.max(1)[1]

                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    valid_accuracies_batches.append(acc_score(targets, output) * len(inputs))

                net.train()
                
            # Append average validation accuracy to list.
            valid_accuracies.append(np.sum(valid_accuracies_batches) / len(validation_dataset))
     
            print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
            print(f"             validation accuracy: {valid_accuracies[-1]}")


print("Finished training.")

epoch = np.arange(len(train_accuracies))
epoch2 = np.arange(len(valid_accuracies))

plt.figure()
plt.plot(epoch, train_accuracies, 'r',epoch2,valid_accuracies,'b')
plt.legend(['Train Accucary','Validation Accuracy'])
plt.xlabel('Updates'), plt.ylabel('Acc')
plt.show()
filepath_fig="/zhome/50/8/147315/DL_Project/figures/Accuracy_plot_%s.png"%(modelname)
plt.savefig(filepath_fig)


# saving to file
filepath="/zhome/50/8/147315/DL_Project/trained_models/"
torch.save(net, filepath + "%s.pt"%(modelname))
torch.save(net.state_dict(), filepath + "%s.pth"%(modelname))
print("done :)")
