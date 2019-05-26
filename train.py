
import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# ------------------------------------------------------------------------------- #
# Define Functions
# ------------------------------------------------------------------------------- #
# Function arg_parser() parses keyword arguments from the command line
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier as int')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
    return args
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    return train_data

def valid_transformer(valid_dir):
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)
    return valid_data


def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    return test_data


def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

# Function check_gpu(gpu_arg) make decision on using CUDA with GPU or CPU
def check_gpu(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

# primaryloader_model(architecture="vgg16") downloads model (primary) from torchvision
def primaryloader_Model(architecture="vgg16"):
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        Model = models.vgg16(pretrained=True)
        Model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else: 
        exec("Model = models.{}(pretrained=True)".format(architecture))
        Model.name = architecture
    
    # Freeze parameters so we don't backprop through them
    for param in Model.parameters():
        param.requires_grad = False 
    return Model

# Function initial_classifier(model, hidden_units) creates a classifier with the corect number of input layers
def initial_classifier(Model, hidden_units):
    # Check that hidden layers has been input
    if type(hidden_units) == type(None): 
        hidden_units = 4096 #hyperparamters
        print("Number of Hidden Layers specificed as 4096.")
    
    # Find Input Layers
    input_features = Model.classifier[0].in_features
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

# Function validation(model, testloader, criterion, device) validates training against testloader to return loss and accuracy
def validation(Model, validloader, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = Model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

# Function network_trainer represents the training of the network model
def network_trainer(Model, trainloader, validloader, testloader, device, 
                  criterion, optimizer, epochs, print_every, steps):
    # Check Model Kwarg
    if type(epochs) == type(None):
        epochs = 5
        print("Number of epochs specificed as 5.")    
 
    print("Training process initializing .....\n")

    # Train Model
    for e in range(epochs):
        running_loss = 0
        Model.train()
         # Technically not necessary, setting this for good measure
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
           
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = Model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                Model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(Model, validloader, testloader, criterion, device)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.3f} | ".format(running_loss/print_every),
                      "Test Loss: {:.3f} | ".format(test_loss/len(testloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
                running_loss = 0
                
                Model.train()

    return Model

#Function validate_model(Model, Testloader, Device) validate the above model on test data images
def validate_Model(Model, trainloader, device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in trainloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
def initial_checkpoint(Model, save_dir, train_data):
    
    print("Our Model:\n\n", Model, '\n')
    print("The statecdict keys:\n\n", Model.state_dict().keys())

    torch.save(Model.state_dict(), 'checkpoint.pth')

    state_dict = torch.load('checkpoint.pth')
    print(state_dict.keys())

    Model.load_state_dict(state_dict)

       
    # Save model at checkpoint
    if type(save_dir) == type(None):
        print("Model checkpoint directory not specified, Model will not be saved.")
    else:
        if isdir(save_dir):
            # Create `class_to_idx` attribute in model
            Model.class_to_idx = train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            image_path = "flowers/test/1/image_06743.jpg"
            # Save checkpoint
            torch.save(checkpoint, 'checkpoint.pth')
            

        else: 
            print("Directory not found, Model will not be saved.")


# =============================================================================
# Main Function
# =============================================================================

# Function main() is where all the above functions are called and executed 
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = train_transformer(train_dir)
    valid_data = valid_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load Model
    Model = primaryloader_Model(architecture=args.arch)
    
    # Build Classifier
    Model.classifier = initial_classifier(Model, 
                                         hidden_units=args.hidden_units)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Send model to device
    Model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(Model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 30
    steps = 0
    

    
    # Train the classifier layers using backpropogation
    trained_Model = network_trainer(Model, trainloader, validloader, testloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is now complete!!")
    
    # Quickly Validate the model
    validate_Model(trained_Model, trainloader, device)
    
    # Save the model
    initial_checkpoint(trained_Model, args.save_dir, train_data)


# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()
    