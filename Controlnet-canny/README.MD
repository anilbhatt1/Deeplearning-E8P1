<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]

## Training and Inferencing Controlnet for Canny Edges Using Google Colab
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* [References](#References)
* [Acronyms](#Acronyms)
* [Overview](#Overview)
* [Approach](#Approach)
* [Attempts and Results](#Attempts-And-Results)
* [License](#license)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  or above
* [Google Colab](https://colab.research.google.com/)

## References

- [ctlnet-orig] → ControlNet original library from lllyasviel
- [ctlnet-train] → Details on how to train a ControlNet to Control Stable Diffusion
- [ctlnet-custom] → ControlNet cloned from lllyasviel. Edited few modules to enable running in colab.
- https://huggingface.co/blog/controlnet → Detailed blog on Using controlnet pretrained model in stable diffusion pipeline in colab

## Acronyms

* SD - Stable Diffusion
* CTLNET - ControlNet
* CTLNET-CLN - CTLNET cloned. Edited few modules to enable it to run in colab. [ctlnet-custom]
* CDS - Custom DataSet

## Overview

- ControlNet is a neural network structure to control diffusion models by adding extra conditions.
- For example, let us say we want to create winged drones flying like birds from a given image.
- SD with CTLNET can help us achieve that as shown below:

Copy a3-inference.png

- In short, CTLNET equips us with the ability to control SD and generate images in the way we desire by passing a control image and prompt.
- In the above example, we are converting the input image having 2 birds by passing the control image (canny edge black & white image) and prompt "A Winged Drone" to get back 2 winged drones flying similarly like the birds in input image.

- Overall architecture of CTLNET is as follows:

Copy CTLNET_Architecture.png

- The "trainable" one (with blue shades) learns our control condition (eg: canny edge, pose etc.). The "locked" one preserves the model.
- Thanks to CTLNET, training with small custom datasets will not destroy the production-ready SD models.

<br>

## Approach
- Training is done based on [ctlnet-train] 
- In this repo, we are attempting to **train CTLNET Canny Edge condition from scratch using a CDS**.
- Objective : Given an input image, its canny edge sketch and a prompt, get back an output image 
    -  Having the object mentioned in the prompt 
    -  With edges resembling the canny edge sketch 
    -  And overall canvas resembling the input image.
- CDS is created in the same format as [Fill50K] dataset.
- CDS is created from 7574 flying bird RGB images that were downloaded from internet (creative common license images).
- CDS is driven via prompt.json - a sample json record shown as below 
    - {"source": "source/0.png", "target": "target/0.png", "prompt": "pale golden rod circle with old lace background"}
    - Here **source** → Control image (in our case Canny Edge Sketch).This was created using cv2.canny.
        
        copy source.jpg
    - **target** → Original image
        
        copy target.jpg
    - **prompt** → A caption explaining the image.This was created using BLIP captioning.

        {"source": "source.jpg", "target": "target.jpg", "prompt": "two birds flying in the sky"}

        copy corresponding prompt
- Notebook dealing with CDS creation is as below. Check the section **Dataset creation for Canny**
    - Give link for S15_ControlNet_V2.ipynb
- Seven attempts were made for arriving at final solution. 
- Details are as follows.

<br>

## Attempts-and-Results

<br>

- **Attempt 1**: 

    - Reference notebook : S15_ControlNet_V2.ipynb
    - Model was created from '/models/cldm_v15.yaml' based on CTLNET-CLN [ctlnet-custom]
    - Trained the whole model from scratch using CDS for 10 epochs with A100 GPUs for batch_size = 8 (947 batches)
    - Training results were bad.
        - Sample created

            Copy a1-sample-created
        - Reconstructed image

            Copy a1-reconstructed-img
        - Control image given as input for conditioning

            Copy a1-control-img
        - Captions supplied

            Copy a1-captions 
    - Used 'gradio_canny2image.py' provided in CTLNET-CLN for inferencing.
    - Preloaded the ckpt file from above step against the model created from '/models/cldm_v15.yaml'
    - Inference results were bad.

        Copy a1-inference
    - **Conclusion** : Realized that training the whole model from scratch requires lot more epochs, data & computing power which is not possible.
    - But to rule out the suspicion that results are bad because CDS is bad, we will train the same way against [Fill50K] dataset.
    - If the results are equally bad, then we can rule out training the whole model from scratch.

<br>

- **Attempt 2** : 

    - Reference notebook : S15_ControlNet_V3.ipynb
    - Created the model same way as in attempt 1.
    - Trained the whole model from scratch using [Fill50K] for 1 epoch with A100 GPUs for batch_size =8.
    - This was done to ensure that CDS is not the culprit. 
    - Training results were equally bad as in attempt 1 confirming that CDS is not to blame.
        - Sample created

            Copy a2-sample-created
        - Reconstructed image

            Copy a2-reconstructed-img
        - Control image given as input for conditioning

            Copy a3-control-img
        - Captions supplied

            Copy a3-captions     
    - **Conclusion**: This confirms that training the whole model from scratch requires lot more epochs, data & computng power which is not possible.
    - Accordingly training the whole model from scratch is dropped.

<br>

- **Attempt 3** : 

    - Reference notebook : S15_ControlNet_Inference_V1.ipynb
    - Model was created from '/models/cldm_v15.yaml' based on CTLNET-CLN [ctlnet-custom]
    - Preloaded the model with canny pretrained weight ckpt file (control_sd15_canny.pth) and inferred with a sample image from CDS. 
        ```
        downloaded_model_path = hf_hub_download(repo_id="lllyasviel/ControlNet",
                                            filename="models/control_sd15_canny.pth",
                                            use_auth_token=True)
        print(downloaded_model_path)

        apply_canny = CannyDetector()

        model = create_model('./models/cldm_v15.yaml').cpu()
        model.load_state_dict(load_state_dict('/content/models/control_sd15_canny.pth', location='cuda'))
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)
        ```
    - Used 'gradio_canny2image.py' provided in CTLNET-CLN for inferencing.
    - This was done to ensure that the entire code works fine in colab when inferred with an established model. 
    - Results were good confirming that code set-up is fine.
    - **Conclusion**: This proves that pretrained model is already good in generating output image conforming to the conditions set via control image and prompt.

        copy a3-inference

<br>

- **Attempt 4** : 

    - Reference notebook : S15_ControlNet_V0.ipynb    
    - Preloaded the model with another SD ckpt file(v1-5-pruned.ckpt) suitable for fine-tuning. 
        ```
        downloaded_model_path = hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5",
                                            filename="v1-5-pruned.ckpt",
                                            use_auth_token=True)
        print(downloaded_model_path)

        input_path = f'/content/models/v1-5-pruned.ckpt'
        output_path = f'/content/models/control_sd15_ini.ckpt'

        model = create_model(config_path='./models/cldm_v15.yaml')

        pretrained_weights = torch.load(input_path)
        ```
    - Trained with [Fill50K] for 1 epoch with A100 GPUs for batch_size = 8  
    - This was done to check whether the entire code including training loop works fine in colab when trained with an established model against a standard dataset like [Fill50K].
    - Results were good.
        - Sample created
            Copy a4-sample-created
        - Reconstructed image
            Copy a4-reconstructed-img
        - Control image given as input for conditioning
            Copy a4-control-img
        - Captions supplied
            Copy a4-captions  
    - **Conclusion** : This proves that pretrained model used further for training CTLNET with CDS provide good results compared to training the same from scratch.

<br>

- Attempt 5 : 

    - Reference notebook : S15_ControlNet_V4.ipynb  
    - Preloaded the model with v1-5-pruned.ckpt.
        ```
        downloaded_model_path = hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5",
                                            filename="v1-5-pruned.ckpt",
                                            use_auth_token=True)
        print(downloaded_model_path)

        input_path = f'/content/models/v1-5-pruned.ckpt'
        output_path = f'/content/models/control_sd15_ini.ckpt'

        model = create_model(config_path='./models/cldm_v15.yaml')

        pretrained_weights = torch.load(input_path)
        ```
    - Initialized the control-model weights (trainable copy) of the model.
        ```
        # Get the state dictionary of the model
        state_dict = model.state_dict()
        # Manually initialize each parameter tensor of control_model (trainable conditioning block)
        i = 0
        for name, param in state_dict.items():
            if 'control_model' in name:
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)
            i += 1  
        # Load the updated state dictionary into the model
        model.load_state_dict(state_dict)
        print(f'Total parms : {i}')
        ```
    - Trained with 10_000 [Fill50K] images for 1 epoch of batch_size = 8 (1250 batches) against A100 High-RAM GPU.
    - This was done to check whether training is happening with established model after initializing the control-model against a standard dataset and how the results look like.
    - Results were good conforming to the conditions set via control image and prompt.
        - Sample created

            Copy a5-sample-created
        - Reconstructed image

            Copy a5-reconstructed-img
        - Control image given as input for conditioning

            Copy a5-control-img
        - Captions supplied

            Copy a5-captions     
    - **Conclusion** : This proves that pretrained model with trainable copy initialized and trained using an established dataset can provide good results conforming to the conditions set via control image and prompt.

<br>

- Attempt 6 :

    - Reference notebook : S15_ControlNet_V5.ipynb  
    - Preloaded the model with v1-5-pruned.ckpt (no canny edge involved so far).
    - Canny inferred a sample CDS image with this pretrained model before initializing the control-model weights.
    - Results were not resembling the canny edge captured.

        copy a6-inference-1
    - Then, initialized the control-model weights (no canny edge involved again so far).
    - Again Canny inferred with the same sample CDS image.
    - Results were again not resembling the canny edge captured.

        copy a6-inference-2

        copy a6-inference-3
    - These were done to ensure that pretrained model (without any canny training) as such cannot provide the results conforming to the conditions (in this case canny edges).
    - **Conclusion** : Results confirm the understanding i.e. model gave output image based on the prompt but output didn't conform to the canny edge control input provided. Hence fine-tuning for specific condition (in this case - canny) is necessary.

<br>

- Attempt 7 : 

    - Reference notebook : S15_ControlNet_V6.ipynb  
    - Preloaded the model with v1-5-pruned.ckpt (no canny edge involved so far).
        ```
        downloaded_model_path = hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5",
                                        filename="v1-5-pruned.ckpt",
                                        use_auth_token=True)

        model = create_model(config_path='./models/cldm_v15.yaml')

        pretrained_weights = torch.load(input_path)
        ```
    - Initialized the control-model weights (trainable copy) of the model.
        ```
        # Get the state dictionary of the model
        state_dict = model.state_dict()

        # Manually initialize each parameter tensor of control_model (trainable conditioning block)
        i = 0
        for name, param in state_dict.items():
            if 'control_model' in name:
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)
            i += 1

        # Load the updated state dictionary into the model
        model.load_state_dict(state_dict)
        print(f'Total parms : {i}')
        ```

    - Trained 7574 flyingbirds images present in CDS & ran 5 epochs for batch_size = 8 (1250 batches) against A100 High-RAM GPU.
    - Results were good.
        - Sample created

            Copy a7-sample-created
        - Reconstructed image

            Copy a7-reconstructed-img
        - Control image given as input for conditioning

            Copy a7-control-img
        - Captions supplied

            Copy a7-captions  
    - Inferred with the same model & results are promising

        copy a7-inference-1

        copy a7-inference-2

        copy a7-inference-3

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/LICENSE.txt
[ctlnet-orig]:https://github.com/lllyasviel/ControlNet
[ctlnet-custom]: https://github.com/anilbhatt1/ControlNet
[ctlnet-train]: https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md
[Fill50K]:https://huggingface.co/lllyasviel/ControlNet/tree/main/training
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555

