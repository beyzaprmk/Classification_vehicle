import torch
import timm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


class ModelTester:
    def __init__(self, model_path, num_classes=2, device=None, image_size=(64,64)):
        # Cihaz
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modeli oluştur ve ağırlıkları yükle
        self.model = timm.create_model('tinynet_e.in1k', pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Test moduna al

        # Transform (train ile aynı olmalı)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Loss fonksiyonu
        self.loss_fn = nn.CrossEntropyLoss()

    def prepare_dataloader(self, data_path, batch_size=16, shuffle=False, num_workers=4):
        dataset = ImageFolder(root=data_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader

    def test(self, dataloader):
        all_labels = []
        all_preds = []
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Testing"):  # tqdm ekledik
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                test_loss += loss.item()
                correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
            
            # Her batch için ilerlemeyi göstermek istersen
                batch_acc = (outputs.argmax(1) == y).type(torch.float).mean().item()
                tqdm.write(f"Batch loss: {loss.item():.4f}, Batch acc: {batch_acc*100:.1f}%")
        test_loss /= num_batches
        accuracy = correct / size
        print(f"Test Error: Accuracy: {accuracy*100:.1f}%, Avg loss: {test_loss:.8f}")

# Örnek kullanım
if __name__ == "__main__":
    tester = ModelTester(model_path='./best_model8.pth')
    testloader = tester.prepare_dataloader(r'C:\Users\beyza.parmak\Desktop\test_data', batch_size=16)
    tester.test(testloader)
