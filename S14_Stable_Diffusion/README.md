
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]

## UNET and VAE
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [Code](#Code)
* [License](#license)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- Code -->
## Code
- **Dataset Used for VAE** : MNIST , Image Resolution : 28x28x1 , CIFAR-10, Image Resolution : 32x32x3
- For Training details of **MNIST VAE**, refer below colab notebook locations:
    - Takes in two inputs:
        - an MNIST image, and
        - its label (one hot encoded vector sent through an embedding layer)
    - Passing label as one-hot vector & concatenating it. Basic classification model, uses custom dataloader. Reference : https://github.com/gkdivya/EVA/blob/main/3_PyTorchNeuralNetwork/MNIST_RandomNumber_Addition.ipynb
File : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S13_UNET_VAE/S13_MNIST_VAE_V0.ipynb
    - VAE using nn.Linear on MNIST. Basic reconstruction not passing one-hot label as input. Reference : https://github.com/lyeoni/pytorch-mnist-VAE
File : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S13_UNET_VAE/S13_MNIST_VAE_V1.ipynb
    - Encoder(maxpool) & Decoder(Transpose2d) model adapted from VAE-MNIST-Experiments.ipynb. Uses Basic reconstruction not passing one-hot label as input to the model. Uses custom dataloader same as that used in S13_MNIST_VAE_V0.ipynb
File : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S13_UNET_VAE/S13_MNIST_VAE_V2.ipynb
    - VAE using nn.Linear on MNIST similar to S13_MNIST_VAE_V1.ipynb. Uses same dataloader as in https://github.com/lyeoni/pytorch-mnist-VAE. Passes one-hot label as input to model and uses torch.add to concatenate image tensor & one-hot label tensor. Gives good results.
File : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S13_UNET_VAE/S13_MNIST_VAE_V3.ipynb
- For Training details of **CIFAR-10 VAE**, refer below colab notebook locations:
    - Takes in two inputs:
        - an MNIST image, and
        - its label (one hot encoded vector sent through an embedding layer)
    - https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S13_UNET_VAE/S13_CIFAR10_VAE_V1.ipynb 
- **Dataset Used for UNET** : OxfordIIITPet
- For Training details of , refer below colab notebook locations:
    - MP + TransposeConv + BCE
        File : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S13_UNET_VAE/S13_UNET_mptrbce.ipynb
    - MP + TransposeConv + Dice Loss
        File : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S13_UNET_VAE/S13_UNET_mptrdice.ipynb
    - StridedConv + TransposeConv + BCE
        File : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S13_UNET_VAE/S13_UNET_strtrbce.ipynb
    - StridedConv + Upsampling + BCE
        File : https://github.com/anilbhatt1/Deep_Learning_EVA8_Phase1/blob/master/S13_UNET_VAE/S13_UNET_strUpsbce.ipynb
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




