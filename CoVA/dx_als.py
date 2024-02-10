import seed
from data_utils import Dataset_Dx

import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

def dx_als(dataset_dir, output_dir, trained_dx_model):
    seed.set()
    
    dataset = Dataset_Dx(dataset_dir, out_filename=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
    
    model = torch.load(trained_dx_model)
    model.eval()

    test_loss = 0
    test_accuracy = 0
    confusion_matrix = torch.zeros(2, 2)

    with torch.no_grad():
        for data, labels, filename in tqdm(dataloader):
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            outputs = outputs.view(-1,outputs.size(-1))
            labels = labels.view(-1)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            score = nn.functional.softmax(outputs, -1)[0,1]
            with open(os.path.join(output_dir,f'log.txt'), 'a') as f:
                print(f"{filename}\nrisk score: {score:.4f}\npred/true: {predicted.data[0]}/{labels.data[0]}\n", file=f)
            test_accuracy += (predicted == labels).sum().item()

            for t, p in zip(labels, predicted):
                confusion_matrix[t.long(), p.long()] += 1

    test_loss /= len(dataloader)
    test_accuracy /= len(dataloader)
    with open(os.path.join(output_dir,'result.txt'), 'a') as f:
        print(f'Loss: {test_loss:.4f}\nAcc: {test_accuracy*100:.4f}%', file=f)
        print('Confusion Matrix:\n', confusion_matrix, file=f)
    print(f'Loss: {test_loss:.4f}\nAcc: {test_accuracy*100:.4f}%')
    print('Confusion Matrix:\n', confusion_matrix)

    plt.figure(figsize=(10,7))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix.png'))
    
    print("Process Completed!")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Argument is missing. (e.g. python3 tokenize.py <dataset directory> <output directory> <trained_dx_model>)")
        sys.exit(1)
    dataset_dir = os.path.join(sys.argv[1])
    output_dir = os.path.join(sys.argv[2])
    trained_dx_model = os.path.join(sys.argv[3])
    dx_als(dataset_dir, output_dir, trained_dx_model)