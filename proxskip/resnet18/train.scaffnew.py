"""Training ResNet-18 on CIFAR-10 dataset using Scaffnew algorithm"""

import torchvision
import torch
import torch.nn as nn
import time
from torchvision.datasets import CIFAR10

# Unique random seed
torch.manual_seed(hash('Xt:!'))

# Train on GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ResNet-specific transforms. Not applying 
# normalization as we are training from scratch
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224,), antialias=True)
])

# Loadeing training part of CIFAR-10
dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms
)


def get_eval_sample(dataset: CIFAR10, per_class: int) -> CIFAR10:
    """Retrieves CIFAR10 slice for evaluation"""
    
    x = [[] for _ in range(10)]
    for i, t in enumerate(dataset.targets):
        x[t].append(i)
        
    x = [
        y[:per_class] for y in x
    ]
    
    i = []
    for k in x:
        i.extend(k)
        
    dataset.data = dataset.data[i]
    dataset.targets = [dataset.targets[x] for x in i]
    
    return dataset

dataset = get_eval_sample(dataset, 300)
print(f"Samples to eval on sync: {len(dataset)}")

def get_models(num: int):
    """Creates model for each of simulated device
    
    Returns a list of length `num` of ResNet-18 models 
    with CIFAR classifier head.
        
    Each model initialized separately with no pre-trained
    weights.

    Args:
        num (int): Number of simulated devices
        
    Returns
        models (list[nn.Module]): initialized models
    """
    models = []
    for _ in range(num):
        resnet = torchvision.models.resnet18()
        resnet.fc = nn.Linear(512, 10)
        resnet.to(device)
        
        models.append(resnet)

    return models


def set_random_weights(model: nn.Module):
    """Iterates over model parameters and set weights
    
    Weights are drawn from Uniform distribution. Each 
    parameter is drawn independently.
    
    Setting happens in-place.
    
    Args:
        model (nn.Module): A model to set weights for
        
    Returns:
        None
    """
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-1, 1)
    


def get_data_splits(num: int) -> list[CIFAR10]:
    """Splits CIFAR-10 into even parts
    
    Each dataset object obtains an independent,
    disjoint set of samples.
    
    Args:
        num (int): A number of devices to create data loaders for
        
    Returns:
        dataloaders (list[CIFAR10]): list of CIFAR10 slices
    """

    datasets = []
    
    for i in range(num):
        resnet_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, transform=transforms
        )
        # Slicing evenly, dropping remainder part (just for simplicity)
        slice_size = len(resnet_dataset) // num
        resnet_dataset.data = resnet_dataset.data[i * slice_size:(i + 1) * slice_size]
        resnet_dataset.targets = resnet_dataset.targets[i * slice_size:(i + 1) * slice_size]
        datasets.append(resnet_dataset)
        
    return datasets


def get_xt(model: nn.Module) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    """Packs all model weights as single vector
    
    The method is used to fed weights to optimizer. Iterates
    over deterministic order of model parameters and packs
    them into single vector
    
    Args:
        model (nn.Module): A model to pack weights of
    
    Returns:
        weights (torch.Tensor): 1D vector for all weights
        shapes (list[tuple[int, ...]]): list of shapes for each parameter
    """
    parameter_names = [k for k, _ in model.named_parameters()]
    parameter_names.sort()
    parameters = []
    shapes = []
    for name in parameter_names:
        parameters.append(model.get_parameter(name).data.flatten())
        shapes.append(model.get_parameter(name).shape)
        
    return torch.cat(parameters), shapes


def pack_grads(model: nn.Module):
    """Packs gradients of model weights as single vector
    
    The method is used to fed gradients to optimizer. Iterates
    over deterministic order of model parameters and packs
    assigned gradients into a single vector
    
    Args:
        model (nn.Module): A model to pack gradients of
    
    Returns:
        weights (torch.Tensor): 1D vector for all weights
        shapes (list[tuple[int, ...]]): list of shapes for each parameter
    """
    parameter_names = [k for k, _ in model.named_parameters()]
    parameter_names.sort()
    parameters = []
    shapes = []
    for name in parameter_names:
        parameters.append(model.get_parameter(name).grad.flatten())
        shapes.append(model.get_parameter(name).shape)
        
    return torch.cat(parameters), shapes   

def sample(dataset: CIFAR10, i: int, size: int):
    """Samples from CIFAR-10 dataset
    
    Args:
        dataset (CIFAR10): A dataset object
        i (int): Index of first sample
        size (int): Size of sample
        
    Returns:
        images (torch.Tensor): stacked images of shape (size, 4, 224, 224)
        labels (torch.LongTensor): labels of shape (size,)
    """
    r = min(i+size, len(dataset))
    data_ = [dataset[j] for j in range(i, r)]
    images_ = [x[0] for x in data_]
    targets_ = [x[1] for x in data_]
    return torch.stack(images_).to(device), \
        torch.tensor(targets_, device=device, dtype=torch.long)


def rsample(dataset: CIFAR10, batch_size: int):
    """Random sample from CIFAR-10 dataset
    
    Args:
        dataset (CIFAR10): A dataset object
        batch_size (int): Size of sample
        
    Returns:
        images (torch.Tensor): stacked images of shape (batch_size, 4, 224, 224)
        labels (torch.LongTensor): labels of shape (batch_size,)
    """
    L = len(dataset) // batch_size
    i = torch.randint(0, L, (1,)).item()
    return sample(dataset, i, batch_size)


def update_weights(model: nn.Module, params: torch.Tensor):
    """Updates weights of the model
    
    The method receives packed params and update
    nn.Module weights accordingly
    
    Args:
        model (nn.Module): Model object to update
        params (nn.Tensor): 1D array of weights
        
    Returns:
        True if updated, otherwise raises error
    """
    assert params.ndim == 1, f"Parameters should be 1D, passed {params.shape}"
    
    parameter_names = [k for k, _ in model.named_parameters()]
    parameter_names.sort()
    shift = 0
    for name in parameter_names:
        numel = model.get_parameter(name).numel()
        shape = model.get_parameter(name).shape
        model.get_parameter(name).data = params[shift:shift + numel].reshape(shape)
        shift += numel
        
    assert torch.allclose(get_xt(model)[0], params), "Parameters failed to update or the order is changed" 
        
    return True


def eval_model(model: nn.Module):
    """Evaluates a given model on the whole dataset"""
    total_loss, n = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            X, y = sample(dataset, i, batch_size)
            y_pred = model.forward(X)
            total_loss += lossfn(y_pred, y).item()
            n += 1
            
    total_loss /= n
    return total_loss

    
# Hyperparameters
learning_rate = 3E-4
learning_rate_decay = 0.99  # To decrease lr with step
num_devices = 4  # number of devices simulated
num_iterations = 25000  # upper bound for iteration
num_com_rounds = 40  # upper bound for com. rounds
p = 1/500  # prob. of calculating prox
batch_size = 64  # size of sample sampled 

lossfn = nn.CrossEntropyLoss()

# Data split for each device
dataloaders = get_data_splits(num_devices)
# Model on each device
models: list[nn.Module] = get_models(num_devices)
# Final, averaged model
final_model = torchvision.models.resnet18()
final_model.fc = nn.Linear(512, 10)
final_model.to(device)

# Setting weights extremely randomly
set_random_weights(final_model)

# The starting point (identical to each model)
x_t_start = get_xt(final_model)[0]
# Size of flattened model parameters
params_size = x_t_start.shape[0]
x_t = [
    x_t_start.clone()
    for _ in range(num_devices)
]

# Setting models to identical values
for i, model in enumerate(models):
    update_weights(model, x_t[i])

update_weights(final_model, x_t_start)

# Initializing control variates. According
# to Line 1 of Algorithm 2
h_t = torch.rand(num_devices, params_size, device=device)
h_t = (h_t - h_t.mean(dim=0))

assert torch.allclose(
    h_t.sum(dim=0), torch.zeros(params_size, device=device), atol=1E-5
), "h_t shoukd sum to 0 vectors across all devices"
    
# Store all losses, step and communication
# counters
all_losses, t, com = [], 0, 0

# Start time of an experiment
start_time = time.time()

# Delta time to estimate speed of a single step
delta = 0


print(f"""
======== Summary ========
Learning rate (gamma):\t\t{learning_rate}
Learning rate decay:\t\t{learning_rate_decay}
Number of devices:\t\t{num_devices}
Maximum iterations:\t\t{num_iterations}
Communication rounds:\t\t{num_com_rounds}
Probability (p):\t\t{p}
Batch size;\t\t\t{batch_size}
Initial loss:\t\t\t{eval_model(final_model):.4f}

""")

try: # Trying for gracefully stopping...
    # Bounded by both iteration and communication rounds
    while t < num_iterations and com < num_com_rounds:
        t += 1
        # Estimate local delta time
        delta_ = time.time()
        # Turn on training
        for model in models:
            model.train(True)
            # Manual .zero_grad()
            for param in model.parameters():
                param.grad = None
            
        # Drawing random samples
        samples = [
            rsample(dl, batch_size)
            for dl in dataloaders
        ]
        
        # Running current models
        y_hat = [
            f(x) for f, (x, _) in
            zip(models, samples)
        ]
        
        # Estimating loss over outputs
        losses: list[torch.Tensor]= [
            lossfn(output, label)
            for (_, label), output
            in zip(samples, y_hat)
        ]
        
        # Propagating loss over with
        # PyTorch frameworkd
        for loss in losses:
            loss.backward()
            
        # Checking whether all gradients
        # are updated
        # assert all(
        #     [all([x.grad is not None for x in y.parameters()]) 
        #      for y in models]
        # ), "Failed to updated gradient"
            
            
        g = [
            pack_grads(m)[0].to(device)
            for m in models
        ]
        
        assert all(
            [g_i.shape[0] == params_size and g_i.ndim == 1 for g_i in g]
        ), f"Gradients are of different size than parameters. {[g_i.shape for g_i in g]}, {params_size}"
            
        # Algorithm 2, Line 6
        x_hat_tp1 = [
            x_t[i] - learning_rate * (g[i] - h_t[i])
            for i in range(num_devices)
        ]
        
        # Deciding on prox evaluation 
        # Algorithm 2, Lines 2, 7
        theta = torch.rand(1).item()
        
        if theta < p:
            # Averaging weights for consensus
            # Algorithm 2, Line 8
            mean_ = torch.stack(x_hat_tp1).mean(dim=0)
            x_tp1 = [
                mean_.clone()
                for _ in range(num_devices)
            ]
            
            # assert all(
            #     [x_tp1[i].shape[0] == params_size for i in range(num_devices)]
            # ), f"Shapes are wrong: {[x_tp1_j.shape for x_tp1_j in x_tp1]}, {params_size}"
            
        else:
            # Skipping prox
            # Algorithm 2, Line 10
            x_tp1 = x_hat_tp1
            
        # Updating control variates
        # Algorithm 2, Line 12
        h_t = [
            h_t[i] + p / learning_rate * (x_tp1[i] - x_hat_tp1[i])
            for i in range(num_devices)
        ]
        
        # Updating models
        for i in range(num_devices):
            update_weights(models[i], x_tp1[i])
            
        x_t = x_tp1
        
        if theta < p:
            com += 1
            # Taking any of identical model
            # after syncronization
            total_loss = eval_model(models[0])
            
            all_losses.append(total_loss)
            print(f"Step: {str(t).zfill(len(str(num_iterations)))}  \tRound: {str(com).zfill(len(str(num_com_rounds)))}"
                f"  \tTotal Loss: {total_loss:.4f}  \t"
                f"MEM: {torch.cuda.memory_allocated() / 10**6:.2f} MB  \t"
                f"Time g={time.time() - start_time:.2f}s, l={delta / t:.2f}s  \t"
                f"LR {learning_rate:.1E}  \t")
            
            learning_rate *= learning_rate_decay
            
        torch.cuda.empty_cache()
        delta_ = time.time() - delta_
        delta += delta_

except KeyboardInterrupt:
    print("Stopping and saving...")

        
torch.save(all_losses, 'resnet18.losses.pt')

weights = torch.zeros(params_size, device=device)
for model in models:
    weights += get_xt(model)[0]
weights /= num_devices

update_weights(final_model, weights)

print(f"\n\nFinal loss: {eval_model(final_model)}")
torch.save(final_model.state_dict(), 'resnet18.weights.pt')