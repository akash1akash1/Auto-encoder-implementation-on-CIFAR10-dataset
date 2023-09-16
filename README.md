# Auto-encoder-implementation-on-CIFAR10-dataset
Auto encoder implementation on CIFAR10 dataset

```bash
class AutoencoderClassifier(nn.Module):
    def __init__(self):
        super(AutoencoderClassifier, self).__init__()
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),           
            nn.Conv2d(24, 48, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Tanh()
        )
        
        self.classifier = nn.Linear(48, 256)
        
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        flattened = encoded.view(-1, 48)
        classified = self.classifier(flattened)
        
        return decoded, classified
```
The autoencoder's encoder section is made up of a number of convolutional and max pooling layers. It uses a tensor of an image as input and performs a number of convolutional layers, followed by max pooling, to increase the number of channels and decrease the spatial dimensions of the picture. It produces a [batch size, 48, 4, 4] encoded tensor.
This is the autoencoder's decoder, which is made up of many transposed convolutional layers. It uses numerous transposed convolutional layers to expand the spatial dimensions of the image while reducing the number of channels, taking the encoded tensor as input. It produces an image tensor that has the following dimensions: [batch size, 3, 32, 32].
This layer of classification applies a linear transformation to the encoded tensor to create a tensor with the shape [batch size, 256]. A softmax function is then applied to the tensor to predict the class probabilities.



```bash
        class_labels = torch.tensor([label for label in labels if label in [1, 3, 5, 7, 9]]) 
        class_outputs_filtered = class_outputs.index_select(0, torch.where(class_labels >= 0)[0])  
        class_labels_filtered = class_labels.index_select(0, torch.where(class_labels >= 0)[0]) 
        loss_class = class_loss(class_outputs_filtered, class_labels_filtered)
```
A tensor of shape (batch size, num classes) called class outputs holds the estimated class probabilities for each input image.
A tuple of two tensors is returned by PyTorch's where() function, one of which contains the indices of the non-zero members and the other of which contains their values. The indices of the labels in class labels that are not negative—i.e., those that are in the set of labels 1, 3, 5, 7, and 9—are found in this line using where().
Then, we utilise index select() to choose only the class outputs rows that match those indices. This results in a tensor of type (num filtered labels, num classes) called class outputs filtered, where num filtered labels is the number of labels from the set of labels that appeared in the current batch (i.e., 1, 3, 5, 7, 9).


## COMPARISON
The accuracy of the CNN model (82%) is more than the Accuracy of the
autoencoder (15.74). This is because we are dealing with a classification
problem.
Another reason why CNN performance is better is due to image
classification. CNN is specifically designed for image recognition tasks.
Another reason why CNN is more accurate than the autoencoder
because the number of layers used in the autoencoder is less(3) than
the CNN (6 layers).
Autoencoder performs well in image denoising or compression but not
suitable for image classification.
