from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights

import global_constants
from data_preprocessing import data_loading


verbose = 2

img_path_list = data_loading.load_data(data_path=global_constants.PREPROCESSED_DATA_PATH, verbose=verbose)
img = img_path_list[0]

# Step 1: Initialize model with the best available weights
weights = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
model = regnet_y_128gf(weights=weights)
model.eval()
# model.train()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
