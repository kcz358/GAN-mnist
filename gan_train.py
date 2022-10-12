from email.mime import image
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class MnistDataset(data.Dataset):
    def __init__(self, mnist_df):
        super().__init__()
        self.db = mnist_df
    
    def __getitem__(self, index):
        return self.db[index]
    
    def __len__(self):
        return len(self.db)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.model(x)


if __name__ == "__main__":
    mnist_df = pd.read_csv("train.csv")
    mnist_df.drop(columns = "label", inplace = True)
    mnist_df = mnist_df.to_numpy(dtype='float32') / 255.0
    
    dataset = MnistDataset(mnist_df)
    
    
    generator = Generator()
    discriminator = Discriminator()
    criterion = nn.BCELoss()
    lr = 0.0001
    batch_size = 128
    nBatches = len(dataset) // batch_size + 1
    num_epochs = 2000
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    data_loader = data.DataLoader(dataset, batch_size = batch_size, shuffle=True)
    
    saving_epochs = [1, 10, 50, 100, 500, 1000, 2000]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    
    for epoch in range(num_epochs):
        for iterations, (real_samples) in enumerate(data_loader):
            real_samples = real_samples.to(device)
            noise = torch.FloatTensor(real_samples.shape[0], 100).to(device)
            noise.data.normal_(0,1)
            fake_samples = generator(noise)
            real_label = torch.ones((real_samples.shape[0],1)).to(device)
            fake_label = torch.zeros((real_samples.shape[0],1)).to(device)
            
            loss_discriminate = criterion(discriminator(torch.cat([real_samples, fake_samples])), 
                                        torch.cat([real_label, fake_label]))
            d_ability = 1 - discriminator(fake_samples).sum()/fake_samples.shape[0]
            
            
            optimizer_discriminator.zero_grad()
            loss_discriminate.backward()
            optimizer_discriminator.step()
            
            noise.data.normal_(0,1)
            fake_samples = generator(noise)
            loss_generator = criterion(discriminator(fake_samples), real_label)
            optimizer_generator.zero_grad()
            loss_generator.backward()
            optimizer_generator.step()
            g_ability = discriminator(fake_samples).sum()/fake_samples.shape[0]
            
            if iterations == 1000:

                print("Prediction ability for discriminator: {}".format(d_ability))
                print("Generation ability for generator: {}".format(g_ability))
                
        if (epoch + 1) in saving_epochs:
            torch.save({
                "epoch" : epoch + 1,
                "generator_state_dict" : generator.state_dict(),
                "discriminator_state_dict" : discriminator.state_dict()
            }, "checkpoints/epoch_{}".format(epoch + 1))