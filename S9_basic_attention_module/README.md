
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]

## Image Classification on CIFAR-10 using basic Attention Model and employing LR Range test & OneCycleLR policy
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Code](#Code)
* [Network Diagram](#Network-Diagram)
* [Loss And Accuracy Plots](#Loss-And-Accuracy-Plots)
* [Misclassified Images](#Misclassified-Images)
* [License](#license)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- Code -->
## Code
- **Dataset Used** : CIFAR-10 , Image Resolution : 32x32x3
- For Training details, refer below colab notebook locations:
    - Completely modularized - only calls main.py in colab notebook
https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S9_basic_attention_module/EVA8_S9_Basic_Self_Attention_V2.ipynb
    - Non-modularized - All functions and classes defined inside colab notebook iteslf
https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S9_basic_attention_module/EVA8_S9_Basic_Self_Attention_V1.ipynb
- **main.py**
    - Main module from which everything else (listed below) were invoked.
- **models.py**    
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/main/src/models.py
	- Model used : **basic_attn_model*
	- **Parameters** :25,680
	- Model trained for 24 epochs with OneCycleLR policy and achieved 38.53% accuracy.
	- Max LR was achieved on 5th epoch.
    - L1 losses were used while training the model. L1_factor=0.0005
	- Optimizer used was ADAM
- **train_loss.py**
    - Location : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/src/train_loss.py
	- Handles the training losses
	- Class : **train_losses**
		- Method **s9_train** is used to train the model.
		- L1 loss is controlled by supplying **L1_factor** while training the models.
		- Training statistics - accuracy, loss - are collected using a **counter** defined in utilities.py
		- Also train accuracy and loss were written to tensorboard using pytorch **Summarywriter** defined as **tb_writer** 
- **test_loss.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/src/test_loss.py
	- Handles the test losses
	- Class : **test_losses**
		- Method **s9_test** is used to test the model.
		- Testing statistics - accuracy, loss - are collected using a **counter** defined in utilities.py. 
		- Additionally misclassified images along with their predicted & actual labels are collected during last epochs.
		- Misclassified images, accuracy and loss were written to tensorboard also using pytorch **Summarywriter**defined as **tb_writer**
- **utilities.py**
	- Location : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/src/utilities.py
	- Class : **stats_collector**
		- stats_collector is used to collect train & test statistics.
		- It also collects misclassified images along with their predicted & actual labels.
	- Function : **ctr**
		- Closure function to collect train & test statistics.
		- It also collects misclassified images along with their predicted & actual labels.
        - **ctr** updates statistics in a global dictionary **counters = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[], 'mis_img':[], 'mis_pred':[], 'mis_lbl':[]}**
    - Class : **Alb_trans**
        - This is a wrapper module to enable image augmentations using Albumentations library.    
        - Following image augmentations were applied on training set:
            - A.Rotate((-5, 5))
            - A.Normalize(mean=channels_mean, std=channels_stdev)
            - A.Sequential([A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=(0.49139968, 0.48215841, 0.44653091)), A.RandomCrop(32, 32)], p=1.0)            
            - A.Cutout(num_holes=1, max_h_size=16, max_w_size=16)
        - Only A.Normalize() was applied on testing set
    - Function : **S9_CIFAR10_data_prep()**
        - This is used to prepare trainloader and testloader 
    - Function : **create_tensorboard_writer**
        - This is used to create tensorboard **SummaryWriter** called **tb_writer** which will be passed on to train_loss and test_loss to collect statistics.
    - Class : **cifar10_plots**
        - Have following functions inside
            - unnormalize_cifar10 : Unnormalizes images by multiplying with stdev & adding mean.
            - plot_cifar10_train_imgs : Plots 5 train images with augmentations applied each of 10 CIFAR-10 classes.
            - plot_cifar10_gradcam_imgs : Plots Gradcam images for layer 1 through 4 for 5 images.
            - plot_cifar10_misclassified : Plots 20 misclassified images identified during testing while last epoch
    - Class : **LRRangeFinder**
        - LRRangeFinder to find max_lr to be supplied to OneCycleLR policy.
    - Function : **plot_onecyclelr_curve**
        - Function to plot onecycleLR curve
        - Accepts list (having learning rates accumulated throughout training) and image save location where graph is stored as jpg file.
        - Learning rates were accumulated for 2352 steps (24 epochs * 98 iterations). Each training epoch has 98 iterations as batch_size used was 512.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555




