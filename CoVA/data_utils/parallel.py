import torch
import torch.nn as nn
from torch.nn.parallel import scatter, parallel_apply

def exists(_value):
    return _value is not None

class DataParallel(nn.Module):
    def __init__(self, net, device_ids=None):
        super(DataParallel, self).__init__()
        self.net = net
        self.device_ids = device_ids if device_ids is not None else list(range(torch.cuda.device_count()))

    def scatter(self, inputs):
        return scatter(inputs, self.device_ids)

    def forward(self, *inputs):
        scattered = self.scatter(inputs[:1])
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        inputs = [scattered[i]+inputs[1:] for i in range(len(self.device_ids))]
        return parallel_apply(replicas, inputs)

    def parallel_loss(self, outputs, targets, criterion, masks=None):
        losses = []
        correct = 0
        total = 0
        if exists(masks):
            masks = self.scatter(masks)
        else:
            masks = [None]*len(outputs)

        for output, target, mask in zip(outputs, self.scatter(targets), masks):
            if exists(mask):
                output, target = output[mask], target[mask]
            losses.append(criterion(output.view(-1,output.size(-1)), target.view(-1)))
            _, predicted = torch.max(output.data, -1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        mean_loss = sum(list(map(lambda loss: loss.item(), losses)))/len(losses)
        accuracy = 100 * correct / total
        return losses, mean_loss, accuracy

    def parallel_backward(self, losses):
        _backward = lambda loss: loss.backward()
        parallel_apply([_backward]*len(self.device_ids), losses)