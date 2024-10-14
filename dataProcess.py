import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        #Normally we used 13000 fake and 700 real videos. for four FF++ 1000 and 700 dataset. 
        for i,file in enumerate(os.listdir(os.path.join(self.root_dir, 'real'))):
            if i<4500:
                self.data.append(os.path.join(self.root_dir, 'real', file))
                self.labels.append(1)
        for i,file in enumerate(os.listdir(os.path.join(self.root_dir, 'fake'))):
            if i < 4500:
                self.data.append(os.path.join(self.root_dir, 'fake', file))
                self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        images = []
        for i, file in enumerate(os.listdir(data_path)):
            if not file.endswith('.txt'):
                image = Image.open(os.path.join(data_path, file))
                if self.transform:
                    image = self.transform(image)
                images.append(image)
        images = torch.stack(images, dim=0)
        landmarks = np.loadtxt(os.path.join(data_path, 'landmarks.txt'))
        landmarks = torch.Tensor(landmarks)
        return (images, landmarks), label
class VideoDataset10(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        #Normally we used 13000 fake and 700 real videos. for four FF++ 1000 and 700 dataset. 
        for i,file in enumerate(os.listdir(self.root_dir)):
            self.data.append(os.path.join(self.root_dir, file))
            self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        images = []
        for i, file in enumerate(os.listdir(data_path)):
            if not file.endswith('.txt'):
                image = Image.open(os.path.join(data_path, file))
                if self.transform:
                    image = self.transform(image)
                images.append(image)
        images = torch.stack(images, dim=0)
        landmarks = np.loadtxt(os.path.join(data_path, 'landmarks.txt'))
        landmarks = torch.Tensor(landmarks)
        return (images, landmarks), label
class VideoDatasetLand(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        #Normally we used 13000 fake and 700 real videos. for four FF++ 1000 and 700 dataset. 
        for i,file in enumerate(os.listdir(os.path.join(self.root_dir, 'real'))):
            if i<2600:
                self.data.append(os.path.join(self.root_dir, 'real', file))
                self.labels.append(1)
        for i,file in enumerate(os.listdir(os.path.join(self.root_dir, 'fake'))):
            if i<2000:
                self.data.append(os.path.join(self.root_dir, 'fake', file))
                self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        images = []
        for file in os.listdir(data_path):
            if not file.endswith('.txt'):
                image = Image.open(os.path.join(data_path, file))
                if self.transform:
                    image = self.transform(image)
                images.append(image)
        images = torch.stack(images, dim=0)
        return images, label
class ImageDataset1(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        for i,image in enumerate( os.listdir(os.path.join(self.root_dir, 'fake'))): 
            self.data.append(os.path.join(self.root_dir,'fake',image))
            self.labels.append(0)
        for i,image in enumerate(os.listdir(os.path.join(self.root_dir, 'real'))):
            self.data.append(os.path.join(self.root_dir,'real',image))
            self.labels.append(1)
                
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        images = []
            
        image = Image.open(data_path)
        if self.transform:
            image = self.transform(image)
        images.append(image)
        images = torch.stack(images,dim=0).squeeze(0) # dim = (1,3,224,224)
        # images = torch.squeeze(images,0) #dim = (3,224,224)
        return images,label
    
class VideoDataset1(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        for i,file in enumerate(os.listdir(os.path.join(self.root_dir, 'real'))):
            if i<1000:
                self.data.append(os.path.join(self.root_dir, 'real', file))
                self.labels.append(1)
        for i,file in enumerate(os.listdir(os.path.join(self.root_dir, 'fake'))):
            if i<3200:
                self.data.append(os.path.join(self.root_dir, 'fake', file))
                self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        images = []
        for file in os.listdir(data_path):
            if not file.endswith('.txt'):
                image = Image.open(os.path.join(data_path, file))
                if self.transform:
                    image = self.transform(image)
                images.append(image)
        images = torch.stack(images, dim=0)
        landmarks = np.loadtxt(os.path.join(data_path, 'landmarks.txt'))
        landmarks = torch.Tensor(landmarks)
        return (images, landmarks), label
        
    
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []   
                            
        for i,video in enumerate(os.listdir(os.path.join(self.root_dir, 'real'))):
            if i < 2500: 
                for image in os.listdir(os.path.join(self.root_dir,'real',video)):
                    if not image.endswith('.txt'):
                        self.data.append(os.path.join(self.root_dir,'real',video,image))
                        self.labels.append(1)
        for i,video in enumerate( os.listdir(os.path.join(self.root_dir, 'fake'))): 
            # if not video.endswith('_c40'):
                if i<2000:
                    for image in os.listdir(os.path.join(self.root_dir,'fake',video)):
                        if not image.endswith('.txt'):
                                self.data.append(os.path.join(self.root_dir,'fake',video,image))
                                self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        images = []
            
        image = Image.open(data_path)
        if self.transform:
            image = self.transform(image)
        images.append(image)
        images = torch.stack(images,dim=0).squeeze(0) # dim = (1,3,224,224)
        # images = torch.squeeze(images,0) #dim = (3,224,224)
        return images,label

class ImageDataset_visualize(Dataset):
    def __init__(self, root_dir, label,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []    
        self.label = label  
        
                            
        for i,video in enumerate(os.listdir(self.root_dir)): 
            for image in os.listdir(os.path.join(self.root_dir,video)):
                if i<1600:
                    if not image.endswith('.txt'):
                        self.data.append(os.path.join(self.root_dir,video,image))
                        if self.label == "real":
                            self.labels.append(1)
                        else:
                            self.labels.append(0)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        images = []
            
        image = Image.open(data_path)
        if self.transform:
            image = self.transform(image)
        images.append(image)
        images = torch.stack(images,dim=0).squeeze(0) # dim = (1,3,224,224)
        # images = torch.squeeze(images,0) #dim = (3,224,224)
        return images,label
class ImageDataset_visualize2(Dataset):
    def __init__(self, root_dir, label,video_name:str,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []    
        self.label = label  
        
                            
        for i,video in enumerate(os.listdir(self.root_dir)): 
            if video.endswith(video_name):
                for image in os.listdir(os.path.join(self.root_dir,video)):
                    if not image.endswith('.txt'):
                        self.data.append(os.path.join(self.root_dir,video,image))
                        if self.label == "real":
                            self.labels.append(1)
                        else:
                            self.labels.append(0)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        images = []
            
        image = Image.open(data_path)
        if self.transform:
            image = self.transform(image)
        images.append(image)
        images = torch.stack(images,dim=0).squeeze(0) # dim = (1,3,224,224)
        # images = torch.squeeze(images,0) #dim = (3,224,224)
        return images,label

class LandmarkDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        for i,file in enumerate(os.listdir(os.path.join(root_dir, 'real'))):
            self.data.append(os.path.join(root_dir, 'real', file, 'landmarks.txt'))
            self.labels.append(1)
        for i,file in enumerate(os.listdir(os.path.join(root_dir, 'fake'))):
            if i<7000:
                self.data.append(os.path.join(root_dir, 'fake', file, 'landmarks.txt'))
                self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]
        landmarks = np.loadtxt(data_path)
        landmarks = torch.Tensor(landmarks)
        return landmarks, label
    


