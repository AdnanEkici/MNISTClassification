import torch

import torch.nn as nn



class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, logits, labels):
        # Calculate your custom CrossEntropyLoss here
        # logits: Tensor of shape (batch_size, num_classes)
        # labels: Tensor of shape (batch_size)

        # Compute the log softmax of the logits
        log_softmax_logits = torch.log_softmax(logits, dim=1)

        # Use negative log likelihood to calculate the loss
        loss = -log_softmax_logits.gather(1, labels.view(-1, 1))

        # Calculate the mean loss over the batch
        loss = torch.mean(loss)

        return loss