# Tree species identification
**Version 1**: classifies tree species from aerial images of 100X100 pixels corresponding to 2x2m patches on the ground.
It also computes the biodiversity of the test images according to three different indexes (Species Richness, Gini Simpson
, and Shannon Wiener).

**Version 2**: identifies the area occupied by the various species from a whole orthomosaic (any image of trees seen
from above and bigger that 100X100 pixels) by using the same model as version 1.

## Usage
### Fine-tune pre-trained models
Training data can be loaded in "tree_classification/data/{dataset_name}", and it must be made known to the algorithm by
setting "input_data_folder = {dataset_name}" in "config.yaml" or by using the input line argument 
"--input_data_folder={dataset_name}". The algorithm expects a single folder containing all the training images,
with file name indicating the species and anything else after a "_" being ignored (e.g. "fir_01.tif")
For data organized in folder and subfolders, there are scripts in "tree_classification/external_scripts" that can be
used to organize the data in the required format.

Example data can be downloaded from this dropbox folder: 
https://www.dropbox.com/scl/fo/hpjrqist8id7h5gvzvni5/h?rlkey=hv30z8bp19qak99nc8acdpwik&dl=0

The script finetune_pretrained_model.py downloads a pre-trained model (from torchvision models) and prepares the data
with standard procedures plus the custom transformations required by the chosen model. Data are split into training,
validation, and test sets, then the algorithm fine-tunes the NN using the training and validation sets, evaluates it on
the test set, and finally displays the results. More than one model can be fine-tuned sequentially. The resulting NNs
are saved in "tree_classification/output/models" with a name chosen by the user (if it is not chosen, the default is
the type of Neural Network used) and an ID to avoid overwriting. Each of them as also a meta_data file associated,
which contains the parameters used to create and train the model and the results obtained during training/fine-tuning
and evaluation. Currently, the supported models from torchvision are: ResNet, RegNet, ConvNeXt, SwinTransformer, but it
is possible to allow more models by adding a file which defines the function "replace_decision_layer" in 
tree_classification/models/pretrained/{model_name}.py.

### Auxiliary scripts
There are other scripts that can be run in the main folder, the list is as follows:
- **finetune_many_models_separate_test**: same as finetune_pretrained_model, but it trains and tests multiple models
sequentially. This script is used when the user wants to employ a separate test set, instead of splitting it from the
training data. the path to the test set must be provided with the parameter "test_data_path = {path_to_test_set}". If
more than one model is trained, it is possible to provide 1 test set (which will be used for all the models), or a
list of test sets (one for each model). The results are saved in "tree_classification/output/models".
- **train_custom_model**: prepares the data, then creates, trains, saves and tests a custom model, and finally displays
the results. With the available data the performance is not comparable to the pre-trained models.
- **test_model**: loads a saved model (from name and ID) and tests it with the test set. Can be used for
both custom and pre-trained fine-tuned models. Can also display the confusion matrix of the test.
- **use_model**: loads a saved model (from name and ID) and uses it to predict the species of a given set of
images. Can be used for both custom and pre-trained fine-tuned models.
- **best_parameters_grid_search**: for each hyperparameter, a list of values can be provided. Performs a grid search
on them (trains and tests models for each possible combination), and outputs the evaluation of each combination of
hyperparameters. The parameter num_tests_for_configuration controls how many models for the same configuration are
trained, tested, and averaged. The results are saved in a json file in "tree_classification/output/parameter_search".
Note that the computation time grows exponentially with the number of hyperparameters with more than 1 value.
- **resume_interrupted_grid_search**: hyperparameter search can require a very long time. If it happens to be
interrupted, the script saves all the intermediate results and with this script the search resumes from that point.
- **show_parameter_search_results**: loads a saved hyperparameter search from
"tree_classification/output/parameter_search" (distinguished by report ID) and displays the results of the grid search
for best hyperparameters that was performed. If only one hyperparameter was searched, it displays a graph of the
results. If two hyperparameters were searched, it displays a 3D graph of the results. If more than two hyperparameters
were searched, it prints on screen the top _n_ best combinations of hyperparameters.
- **show_difficult_cases**: applies a trained/fine-tuned model on a set of images. Then it displays the worst _n_ cases
(where the model was most confident in a wrong classification) along with the model's output, one by one so that users
can manually assess the model's behaviour in the small subset of wrongly classified images.
- **show_confusion_matrix**: loads a saved model's meta-data file (from name and ID) and displays the confusion matrix
of the test that was performed on it after it was trained/fine-tuned. All the information is already stored in the
meta-data, so there is no need to redo the test, or even load the model.
- **compute_biodiversity**: can operate in two different ways: 1) calculates the biodiversity of a set of images, or 2)
calculates the biodiversity of the predictions given by a model over a set of images.

Parameters for the scripts can be modified in the file "config.yaml" or provided via command line argument. There are
also some script-specific parameters that can be modified directly inside some of the scripts.

### Analyze orthomosaic
The script analyse_orthomosaic.py automatically applies the model to a whole orthomosaic and creates a heatmap of the 
species distribution. It uses a model trained on 100X100 pixels patches with the first script described, and it expects
the orthomosaic to be in the folder "tree_classification/data/orthomosaic/{your_orthomosaic_folder}" with the name
"orthomosaic". The chosen folder must be specified to the script through the parameter
"img_folder = {your_orthomosaic_folder}" (modify config.yaml or use command line argument). The extra folder layer is
needed because it is possible to add in the same folder the expected output to evaluate the algorithm.
The output of this script is an image file for each species with the same shape as the orthomosaic, where each pixel
is colored only if the NN thinks there is that species in the corresponding position in the orthomosaic. If a pixel
is colored, its degree of transparency represent the certainty with which the algorithm classified it. The output is
saved in "tree_classification/output/orthomosaic/{your_orthomosaic_folder}". The output consists of one image for each
species, an extra image for unknown species/entities, a legend for the colors used, and a copy of the source image used
. The copy is provided only in the case that a subset of the orthomosaic was used, and it is named "subset_img".
If the input folder contains the expected output, the script also computes accuracy, precision, recall, and f1 score of
the algorithm and shows it.
