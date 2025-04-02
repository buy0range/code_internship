import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 超参数设置
batch_size = 128
latent_dim = 100
num_classes = 10
img_size = 28
img_shape = (1, img_size, img_size)
lr = 0.0002
epochs_cnn = 5
epochs_gan = 20

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

# 定义CNN分类器
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练CNN分类器
cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

print("Training CNN Classifier...")
for epoch in range(epochs_cnn):
    for i, (images, labels) in enumerate(train_loader):
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs_cnn}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 定义条件GAN生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# 定义条件GAN判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(784 + num_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        d_in = torch.cat((self.label_emb(labels), img_flat), -1)
        validity = self.model(d_in)
        return validity

# 初始化GAN
generator = Generator()
discriminator = Discriminator()
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 训练条件GAN
print("\nTraining CGAN...")
for epoch in range(epochs_gan):
    for i, (imgs, labels) in enumerate(train_loader):
        batch_size = imgs.shape[0]
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)
        
        # 训练判别器
        optimizer_D.zero_grad()
        
        real_loss = adversarial_loss(discriminator(imgs, labels), valid)
        
        z = torch.randn(batch_size, latent_dim)
        gen_labels = torch.randint(0, num_classes, (batch_size,))
        gen_imgs = generator(z, gen_labels)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # 训练生成器
        optimizer_G.zero_grad()
        
        gen_labels = torch.randint(0, num_classes, (batch_size,))
        gen_imgs = generator(z, gen_labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{epochs_gan}] [Batch {i}/{len(train_loader)}] "
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

# 测试流程
def generate_digit(label):
    z = torch.randn(1, latent_dim)
    label = torch.tensor([label], dtype=torch.long)
    gen_img = generator(z, label).detach().squeeze().numpy()
    gen_img = 0.5 * gen_img + 0.5  # 反归一化
    plt.imshow(gen_img, cmap='gray')
    plt.axis('off')
    plt.show()

# 使用示例
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=1, shuffle=True)

# 随机选择一个测试样本
test_img, true_label = next(iter(test_loader))
predicted_label = torch.argmax(cnn(test_img), 1).item()

print(f"True label: {true_label.item()}, Predicted label: {predicted_label}")
print("Generating new image with predicted label...")
generate_digit(predicted_label)
