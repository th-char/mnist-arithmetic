# MNIST Arithmetic Datasets

This repo contains three datasets to evaluate 
ML models. The three variants of the dataset 
are 
1. **MNIST_Add:** Given three MNIST digits, output 
their sum
2. **MNIST_Mult:** Given three MNIST digits, output 
their sum
3. **MNIST_MA:** (Stands for Mult-Add), given three 
MNIST digits output the product of the first two 
plus the final one.

The datasets can be created in the following way:
```
mnist_add_dataset = MNIST_Add("path/to/save/mnist/images")
...
```
Each element of the dataset has four elements, the
first is the id of the element (useful when looking 
at elements that have been shuffled in a dataloader),
the second is the raw MNIST images, the third 
are the latent labels for each image and the final 
label is the downstream answer to the task.

Example usage:
```
dataloader_train = DataLoader(
  MNIST_Mult(MNIST_PATH), batch_size=16)
dataloader_val = DataLoader(
  MNIST_Mult(MNIST_PATH, dataset_type="val"), batch_size=16)

def batches_to_examples(*batch_x):
  """ Turns a list of batches into a batch of lists"""
  N, *R = tensors[0].shape

  interleaved_tensor = torch.hstack([t.unsqueeze(1) for t in tensors])
  return interleaved_tensor.view(N * len(tensors), *R)

for i, (_, xs, _, y) in enumerate(dataloader):
    model.train()
    opt.zero_grad()
    
    nn_x = batches_to_examples(*xs).reshape(-1, 3, 28, 28)
    num_examples = nn_x.shape(0)

    nn_out = model(nn_x)
    nn_y = model.activation(nn_out).view(num_examples, -1)
        
    loss = loss_fn(nn_out, y)
    
    running_nn_acc += (nn_y.argmax(axis=1) == y).float().sum()

    loss.backwards()
    opt.step()
```
