import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from dataset import pointDataset
from model import generator, discriminator
from torchvision import transforms
from torch.autograd import Variable

def gen_update(noise, generator, discriminator, genopt):
    output = generator(noise)
    real_output = discriminator(output)

    ones = Variable(torch.FloatTensor(np.ones(real_output.size(0), dtype=np.int)))

    adv_loss = nn.functional.binary_cross_entropy(real_output, ones)

    adv_loss.backward()
    genopt.step()

def dis_update(noise, point, generator, discriminator, disopt):
    output = generator(noise)
    fake_output = discriminator(output)
    real_output = discriminator(point)

    ones = Variable(torch.FloatTensor(np.ones(real_output.size(0), dtype=np.int)))
    zeros = Variable(torch.FloatTensor(np.zeros(fake_output.size(0), dtype=np.int)))

    real_adv_loss = nn.functional.binary_cross_entropy(real_output, ones)
    fake_adv_loss = nn.functional.binary_cross_entropy(fake_output, zeros)

    adv_loss = real_adv_loss + fake_adv_loss
    adv_loss.backward(retain_graph=True)
    disopt.step()


# define three guassian distribution
mu = [[-8, 6], [8, 6], [6, -6]]
cov_1 = [[1, 0], [0, 1]]
cov_2 = [[1, 0], [0, 1]]
cov_3 = [[1, 0], [0, 1]]


x_1, y_1 = np.random.multivariate_normal(mu[0], cov_1, 3200).T
x_2, y_2 = np.random.multivariate_normal(mu[1], cov_2, 3200).T
x_3, y_3 = np.random.multivariate_normal(mu[2], cov_3, 3200).T

point_cluster_1 = np.vstack((x_1, y_1)).T
point_cluster_2 = np.vstack((x_2, y_2)).T
point_cluster_3 = np.vstack((x_3, y_3)).T
training_data = np.concatenate([point_cluster_1,point_cluster_2,point_cluster_3],0)
training_data = torch.from_numpy(training_data)
index = np.arange(training_data.shape[0])

# Define model
generator = generator()
discriminator = discriminator()

# Loading data
transformation = transforms.Compose([transforms.ToTensor])
pointData = pointDataset(training_data, index, transform=None)
train_loader = torch.utils.data.DataLoader(pointData, batch_size=64, shuffle=False)

# Setting optimizer
generatorOptimizor = torch.optim.Adam(generator.parameters(), lr = 0.001)
discriminatorOptimizor = torch.optim.Adam(discriminator.parameters(), lr = 0.001)
iterations = 0
fix_noise = torch.randn(64, 2)
for epoch in range(0, 1000):
    discriminator.train()
    for it, (point,_) in enumerate(train_loader):
        generator.train()
        noise = torch.randn(64, 2)
        noise = Variable(noise)
        point = Variable(point.float())
        fix_noise = Variable(fix_noise)

        dis_update(noise, point, generator, discriminator, discriminatorOptimizor)
        gen_update(noise, generator, discriminator, generatorOptimizor)


        if (iterations+1) % 500 == 0:
            generator.eval()
            generate_point = generator(fix_noise)
            x = generate_point.data[:, 0].numpy()
            y = generate_point.data[:, 1].numpy()
            x_real = training_data.data[:, 0].numpy()
            y_real = training_data.data[:, 1].numpy()
            plt.plot(x_real, y_real, 'bx')
            plt.plot(x, y, 'ro')
            plt.axis('equal')
            plt.show()

        iterations += 1



