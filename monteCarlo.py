import sys
import numpy as np
import torch
import torch.nn as nn


class MonteCarloDropoutModel():
    def __init__(self, model, data_loader, n_samples, classes=1, forward_passes=20, _verb=False, adjust_dropout=0.1):
        model.adjust_dropout(adjust_dropout)
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.n_samples = n_samples
        self.classes = classes
        self.forward_passes = forward_passes
        self._verb = _verb
    
    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    #Verb prints out the same index of output for every pass.          
    def get_monte_carlo_predictions(self):
        dropout_predictions = np.empty((0, self.n_samples))
        sigmoid = nn.Sigmoid()

        for fp in range(self.forward_passes):
            print("pass:", fp)
            predictions = np.empty((0,))
            self.model.train()
            self.enable_dropout()
            verb = self._verb
            for _, (image, label) in enumerate(self.data_loader):
                image = image.unsqueeze(1)
                label = label.float().unsqueeze(1)
                if verb:
                    print("image shape", image.shape, ", fp_n=",fp)
                    print("Training: ", self.model.training)
                with torch.no_grad():
                    output = self.model(image)
                    output = sigmoid(output).squeeze(1)  # shape (n_samples,)
                    if verb:
                        print("output shape", output.shape, output[2])
                predictions = np.append(predictions, output.cpu().numpy())
                verb = False

            dropout_predictions = np.vstack((dropout_predictions, predictions))        
            # dropout predictions - shape (forward_passes, n_samples)

        # Calculating mean across multiple MCD forward passes 
        mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples,)

        # Calculating standard deviation across multiple MCD forward passes 
        std = np.std(dropout_predictions, axis=0)  # shape (n_samples,)

        epsilon = sys.float_info.min

        # Calculating entropy for binary classification
        entropy = -(mean * np.log(mean + epsilon) + (1 - mean) * np.log(1 - mean + epsilon))  # shape (n_samples,)

        # Calculating mutual information for binary classification
        expected_entropy = -np.mean(dropout_predictions * np.log(dropout_predictions + epsilon) +
                                    (1 - dropout_predictions) * np.log(1 - dropout_predictions + epsilon), axis=0)
        mutual_info = entropy - expected_entropy  # shape (n_samples,)

        return mean, std, mutual_info, dropout_predictions