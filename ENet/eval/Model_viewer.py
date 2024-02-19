<<<<<<< HEAD
import torch

# Load the model from the .pth file
model_path = 'state_dict15.pth'
model = torch.load(model_path)
keys = [key for key in model]
print(keys)
# print(model)
# print(model.items())
=======
import torch

# Load the model from the .pth file
model_path = 'state_dict15.pth'
model = torch.load(model_path)
keys = [key for key in model]
print(keys)
# print(model)
# print(model.items())
>>>>>>> 3f06b50c51741a130c22f30daab1ef2b6285ee8c
