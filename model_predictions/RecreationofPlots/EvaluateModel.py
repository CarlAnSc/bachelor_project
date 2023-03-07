import matplotlib.pyplot as plt
import pandas as pd
import torch

from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm
from torchvision import datasets, transforms

from src.imagenet_x.evaluate import ImageNetX, get_vanilla_transform
from src.imagenet_x import FACTORS, plots

imagenet_val_path = '../../data/ImageNetVal'
transforms = get_vanilla_transform()
dataset = ImageNetX(imagenet_val_path, transform=transforms)    

torch.manual_seed(420)

# Load the model
#Default ResNet18.
modelname= 'resnet18'
model = resnet18(weights=ResNet18_Weights.DEFAULT)

#Dino VIT:
#model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

device = 0
batch_size = 128
num_workers = 4
# Evaluate model on ImageNetX using simple loop
model.eval()
model.to(device)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
)
names = loader.dataset.samples

preds = []
probs = []
correct = 0
total = 0

with torch.no_grad():
    for data, target, annotations in tqdm(loader, desc='Evaluating on ImageNetX'):
        data, target = data.to(device), target.to(device)
        annotations = annotations.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        prob = output.softmax(dim=1)
        top_p = prob.topk(1, dim=1)[0]
        mask = pred.eq(target.view_as(pred)).to(device)
        correct += annotations[mask,:].to(device).sum(dim=0)
        total += annotations.to(dtype=torch.int).sum(dim=0)
        preds.append(pred.cpu())
        probs.append(top_p.cpu())

preds1 = torch.cat(preds).numpy()
probs1 = torch.cat(probs).numpy()

namesdf = pd.DataFrame(names, columns=['name', 'index'])
namesdf['name'] = namesdf['name'].str.split('/', expand=True)[6].T
namesdf = namesdf.drop(columns=['index'])
df = pd.DataFrame([preds1, probs1], index=['pred', 'probs']).T
df['probs'] = df['probs'].astype(float)
df = pd.concat([namesdf, df], axis=1)

df.to_csv(modelname+'.csv', index=False)

# Compute accuracies per factor
factor_accs = (correct/total).cpu().detach().numpy()
results = pd.DataFrame({'Factor': FACTORS, 'acc': factor_accs}).sort_values('acc', ascending=False)

# Compute error ratios per factor
results['Error ratio'] = (1 - results['acc']) / (1-(correct.sum()/total.sum()).item())

# Plot results
plots.plot_bar_plot(results, x='Factor', y='Error ratio')
plt.savefig(modelname+'_error_ratio.png')
plt.show()