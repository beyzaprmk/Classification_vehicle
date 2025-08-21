import torch 
import torchvision  # type: ignore
import torchvision.datasets as datasets # type: ignore
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder  # type: ignore
import torch.nn as nn # type: ignore
import os 
from timm.data import create_transform # type: ignore
import timm # type: ignore
import torch.optim as optim # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
PATH = './last.pth'
def to_rgb(img):
    if img.mode in ("P", "RGBA"):
        return img.convert("RGBA").convert("RGB")
    return img.convert("RGB")
transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(p=0.5) , # %50 ihtimalle yatay çevir
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = datasets.ImageFolder(root=r'\\xxx.xx.xx.xx\BRK-Ortak\Genel_Mudurluk_Ortak_Alan\DosyaPaylasimi\AI_Labeling\vehicle_text_classification', transform=transform)
val_data = datasets.ImageFolder(root=r'C:\Users\beyza.parmak\Desktop\val_data', transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True,num_workers=8, pin_memory=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False)

model = timm.create_model('caformer_b36.sail_in22k_ft_in1k', pretrained=True, num_classes=2)  
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
best_loss = float('inf')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
def validate_loop(dataloader, model, criterion):
    all_preds = []
    all_labels = []
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, accuracy

def train_loop(dataloader,val_loader, model, optimizer, criterion, best_loss):
    for epoch in range(20): 
        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
           
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)   
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            

        train_loss = running_loss / len(trainloader)  
        train_accuracy = 100 * correct / total  
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_loop(val_loader, model, criterion)
        print(f"Epoch [{epoch+1}] "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")
        torch.save(model.state_dict(), PATH)
       
        if  val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "./best_model9.pth")

def main():
    train_loop(trainloader,val_loader, model, optimizer, loss_fn, best_loss)
    print('Finished Training')
    
if __name__ == "__main__":
    main()