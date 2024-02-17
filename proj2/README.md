# 16-825 Assignment 2: Single View to 3D

Goals: In this assignment, you will explore the types of loss and decoder functions for regressing to voxels, point clouds, and mesh representation from single view RGB input. 

## 0. Setup

Please download and extract the dataset from the shared Google Drive. 

We provide two versions for the dataset, 
* [here](https://drive.google.com/file/d/1fRaJevvaW_B_ZLb67dqARzJvk7gYGMbG/view?usp=sharing) for a single-class dataset which contains one class of chair. Total size 7.3G after unzipping.

* [here](https://drive.google.com/file/d/1SpJycuBfSmOjVtmbYOs3vEIF83AMIzgD/view?usp=sharing) for an extended version which contains three classes, chair, plane, and car.  Total size 48G after unzipping.

Note: if you are working on remote machine, you can use `gdown https://drive.google.com/uc?id=<file_id>` command in terminal to download the zip file.

After unzipping, set the appropriate path references in `dataset_location.py` file [here](dataset_location.py).

The extended version is required for Q3.3; for other parts, using single-class version is sufficient.

Make sure you have installed the packages mentioned in `requirements.txt`.
This assignment will need the GPU version of pytorch.

## 1. Exploring loss functions
This section will involve defining a loss function, for fitting voxels, point clouds and meshes.

### 1.1. Fitting a voxel grid (5 points)
In this subsection, we will define binary cross entropy loss that can help us <b>fit a 3D binary voxel grid</b>.
Define the loss functions `voxel_loss` in [`losses.py`](losses.py) file. 
For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 

Visualize the optimized voxel grid along-side the ground truth voxel grid using the tools learnt in previous section.

### 1.2. Fitting a point cloud (5 points)
In this subsection, we will define chamfer loss that can help us <b> fit a 3D point cloud </b>.
Define the loss functions `chamfer_loss` in [`losses.py`](losses.py) file.
<b>We expect you to write your own code for this and not use any pytorch3d utilities. You are allowed to use functions inside pytorch3d.ops.knn such as knn_gather or knn_points</b>

Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 

Visualize the optimized point cloud along-side the ground truth point cloud using the tools learnt in previous section.

### 1.3. Fitting a mesh (5 points)
In this subsection, we will define an additional smoothening loss that can help us <b> fit a mesh</b>.
Define the loss functions `smoothness_loss` in [`losses.py`](losses.py) file.

For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 

Visualize the optimized mesh along-side the ground truth mesh using the tools learnt in previous section.

## 2. Reconstructing 3D from single view
This section will involve training a single view to 3D pipeline for voxels, point clouds and meshes.
Refer to the `save_freq` argument in `train_model.py` to save the model checkpoint quicker/slower. 

We also provide pretrained ResNet18 features of images to save computation and GPU resources required. Use `--load_feat` argument to use these features during training and evaluation. This should be False by default, and only use this if you are facing issues in getting GPU resources. You can also enable training on a CPU by the `device` argument. Also indiciate in your submission if you had to use this argument. 

### 2.1. Image to voxel grid (20 points)
In this subsection, we will define a neural network to decode binary voxel grids.
Define the decoder network in [`model.py`](model.py) file for `vox` type, then reference your decoder in [`model.py`](model.py) file

Run the file `python train_model.py --type 'vox'`, to train single view to voxel grid pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D voxel grid and a render of the ground truth mesh.

### 2.2. Image to point cloud (20 points)
In this subsection, we will define a neural network to decode point clouds.
Similar as above, define the decoder network in [`model.py`](model.py) file for `point` type, then reference your decoder in [`model.py`](model.py) file.

Run the file `python train_model.py --type 'point'`, to train single view to pointcloud pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth point cloud and predicted  point cloud in `eval_model.py` file using:
`python eval_model.py --type 'point' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D point cloud and a render of the ground truth mesh.


### 2.3. Image to mesh (20 points)
In this subsection, we will define a neural network to decode mesh.
Similar as above, define the decoder network in [`model.py`](model.py) file for `mesh` type, then reference your decoder in [`model.py`](model.py) file.

Run the file `python train_model.py --type 'mesh'`, to train single view to mesh pipeline, feel free to tune the hyperparameters as per your need. We also encourage the student to try different mesh initializations (i.e. replace `ico_sphere` by other shapes).


After trained, visualize the input RGB, ground truth mesh and predicted mesh in `eval_model.py` file using:
`python eval_model.py --type 'mesh' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted mesh and a render of the ground truth mesh.

### 2.4. Quantitative comparisions(10 points)
Quantitatively compare the F1 score of 3D reconstruction for meshes vs pointcloud vs voxelgrids.
Provide an intutive explaination justifying the comparision.

For evaluating you can run:
`python eval_model.py --type voxel|mesh|point --load_checkpoint`


On your webpage, you should include the f1-score curve at different thresholds for voxelgrid, pointcloud and the mesh network. The plot is saved as `eval_{type}.png`.

### 2.5. Analyse effects of hyperparams variations (10 points)
Analyse the results, by varying a hyperparameter of your choice.
For example `n_points` or `vox_size` or `w_chamfer` or `initial mesh (ico_sphere)` etc.
Try to be unique and conclusive in your analysis.

### 2.6. Interpret your model (15 points)
Simply seeing final predictions and numerical evaluations is not always insightful. Can you create some visualizations that help highlight what your learned model does? Be creative and think of what visualizations would help you gain insights. There is no `right' answer - although reading some papers to get inspiration might give you ideas.


## 3. Exploring other architectures / datasets. (Choose at least one! More than one is extra credit)

### 3.1 Implicit network (10 points)
Implement an implicit decoder that takes in as input 3D locations and outputs the occupancy value. Start with a simple implementation of a network that predicts the occupancy given the image feture and a 3d coordinate as input. You will need to create a meshgrid of 32x32x32 in the normalized coordinate space of (-1,1)^3 to predict the full occupancy output. 

Some papers for inspiration [[1](https://arxiv.org/abs/2003.04618),[2](https://arxiv.org/abs/1812.03828)]

### 3.2 Parametric network (10 points)
Implement a parametric function that takes in as input sampled 2D points and outputs their respective 3D points. 
Some papers for inspiration [[1](https://arxiv.org/abs/1802.05384),[2](https://arxiv.org/abs/1811.10943)]

### 3.3 Extended dataset for training (10 points)
In the extended dataset, we provide a `split_3c.json` file that specifies the train/test split for the extended dataset.

Update `dataset_location.py` so that we train the 3D reconstruction model on an extended dataset containing three classes (chair, car, and plane). Choose at least one of three models (voxel, point cloud, or mesh) to train and evaluate.

After training, compare the quantitative and qualitative results of "training on one class" VS "training on three classes". Explain your thoughts and analysis.

(Hints: for example, given the same testing samples in `chair` class, how does F1 score change comparing "training on one class" and "training on three classes"? How does the 3D consistency / diversity of the output samples change?)
