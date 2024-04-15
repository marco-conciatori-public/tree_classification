# Tree species identification
**Version 1**: classifies tree species from aerial images of 100X100 pixels corresponding to 2x2m patches on the ground.
It also compute the biodiversity of the test images according to three different indexes (Species Richness, Gini Simpson
, and Shannon Wiener).

**Version 2**: identifies the area occupied by the various species from a whole orthomosaic (any image of trees seen
from above and bigger that 100X100 pixels) by using the same model as version 1.

## Usage
### Fine-tuning pre-trained models
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
are saved in "output/models" with a name chosen by the user (if it is not chosen, the default is the type of Neural
Network used) and an ID to avoid overwriting. Each of them as also a meta_data file associated, which contains the
parameters used to create and train the model and the results obtained during training/fine-tuning and evaluation.

### Auxiliary scripts
There are other scripts that can be run in the main folder, the list is as follows:
- **train_custom_model**: prepares the data, then creates, trains, saves and tests a custom model, and finally displays
the results, but with the available data the performance is not comparable to the pre-trained models.
- **load_and_test_model**: loads a saved model (from name and ID) and just tests it with the test set. Can be used for
both custom and pre-trained fine-tuned models, can also display the confusion matrix of the test.
- **load_and_use_model**: loads a saved model (from name and ID) and uses it to predict the species of a given set of
images. Can be used for both custom and pre-trained fine-tuned models.
- **best_parameters_grid_search**: for each hyperparameter, a list of values can be provided. Performs a grid search
on them (trains and tests models for each possible combination), and outputs the evaluation of each combination of
parameters. The parameter num_tests_for_configuration controls how many models for the same configuration are trained,
tested, and averaged. The results are saved in a json file in "output/parameter_search".
- **resume_interrumped_grid_search**: hyperparameter search can require a very long time. If it happens to be
interrupted, the script saves all the intermediate results and with this script the search resumes from that point.
- **show_parameter_search_results**: loads a saved parameter search from "output/parameter_search" (distinguished by
report ID) and displays the results of the grid search for best hyperparameters that was performed. If only one
parameter was searched, it displays a graph of the results. If two parameters were searched, it displays a 3D graph of
the results. If more than two parameters were searched, it prints on screen the top _n_ best combinations of parameters.
- **show_difficult_cases**: use a trained/fine-tuned model on a set of images. Then it displays the worst _n_ cases
(where the model was most confident in a wrong classification) along with the model's output, one by one so that users
can manually assess the model's behaviour in the small subset of wrongly classified images.
- **show_confusion_matrix**: loads a saved model's meta-data file (from name and ID) and displays the confusion matrix
of the test that was performed on it after it was trained/fine-tuned. All the information is already stored in the
meta-data, so there is no need to redo the test, or even load the model.

Parameters for the scripts can be modified in the file "config.yaml" or provided via command line argument. There are
also some script-specific parameters that can be modified directly inside some of the scripts.

### Analyze orthomosaic
- **analyze_orthomosaic**: automatically apply the model to a whole orthomosaic and creates an heatmap of the 
species distribution. **Not fully implemented yet**.