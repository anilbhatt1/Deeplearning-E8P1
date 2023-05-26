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

 ![a3-inference](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/d44ea153-e4dd-404c-82c3-020b2457ade3)
 
- In short, CTLNET equips us with the ability to control SD and generate images in the way we desire by passing a control image and prompt.
- In the above example, we are converting the input image having 2 birds by passing the control image (canny edge black & white image) and prompt "A Winged Drone" to get back 2 winged drones flying similarly like the birds in input image.

- Overall architecture of CTLNET is as follows:

![CTLNET_Architecture](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/02dbaad1-70d7-4cef-9f0f-bf84f1ce22e0)

- The "trainable" one (with blue shades) learns our control condition (eg: canny edge, pose etc.). The "locked" one preserves the model.
- Thanks to CTLNET, training with small custom datasets will not destroy the production-ready SD models.

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
        
        ![source](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/90ec02f4-b32e-40a9-8fcf-6b34455f2654)
        
    - **target** → Original image
        
        ![target](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/4a99e4a6-fe45-427e-97a6-55e441fdd435)
        
    - **prompt** → A caption explaining the image.This was created using BLIP captioning.

        {"source": "source.jpg", "target": "target.jpg", "prompt": "two birds flying in the sky"}

- Notebook dealing with CDS creation is as below. Check the section **Dataset creation for Canny**
    - [S15_ControlNet_V2.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/Controlnet-canny/S15_ControlNet_V2.ipynb)
- Seven attempts were made for arriving at final solution. 
- Details are as follows.

## Attempts-and-Results

- **Attempt 1**: 

    - Reference notebook : [S15_ControlNet_V2.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/Controlnet-canny/S15_ControlNet_V2.ipynb)
    - Model was created from '/models/cldm_v15.yaml' based on CTLNET-CLN [ctlnet-custom]
    - Trained the whole model from scratch using CDS for 10 epochs with A100 GPUs for batch_size = 8 (947 batches)
    - Training results were bad.
        - Sample created

            ![a1-sample-created](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/4a75135c-0d81-4ec2-8561-323185a2efe0)
        - Reconstructed image
            
            ![a1-reconstructed-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/5817a2aa-9d80-40ec-8504-e5a90c59a536)
        - Control image given as input for conditioning
            
            ![a1-control-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/0bf1fd6e-3fe0-4505-b0f1-477efae799a0)
        - Captions supplied
            
            ![a1-captions](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/9cad7040-39e0-4297-b752-d4c5c1b44716)
    - Used 'gradio_canny2image.py' provided in CTLNET-CLN for inferencing.
    - Preloaded the ckpt file from above step against the model created from '/models/cldm_v15.yaml'
    - Inference results were bad.

        ![a1-inference](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/bae711b7-7ea8-40cd-b5e1-fceddf735a00)
    - **Conclusion** : Realized that training the whole model from scratch requires lot more epochs, data & computing power which is not possible.
    - But to rule out the suspicion that results are bad because CDS is bad, we will train the same way against [Fill50K] dataset.
    - If the results are equally bad, then we can rule out training the whole model from scratch.

- **Attempt 2** : 

    - Reference notebook : [S15_ControlNet_V3.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/Controlnet-canny/S15_ControlNet_V3.ipynb)
    - Created the model same way as in attempt 1.
    - Trained the whole model from scratch using [Fill50K] for 1 epoch with A100 GPUs for batch_size =8.
    - This was done to ensure that CDS is not the culprit. 
    - Training results were equally bad as in attempt 1 confirming that CDS is not to blame.
        - Sample created

            ![a2-sample-created](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/73c67678-d246-4a99-9af7-9e9d618e41ae)
        - Reconstructed image
            
            ![a2-reconstructed-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/b54f6164-071a-4ba0-9949-52af728270e7)
        - Control image given as input for conditioning

            ![a2-control-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/6dfbd7e8-1d8d-4f9d-8eb1-5ba1bcaeb20e)
        - Captions supplied
            ![a2-captions](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/0c504a73-6abc-4b1c-9e4b-b6c268054945)  
    - **Conclusion**: This confirms that training the whole model from scratch requires lot more epochs, data & computng power which is not possible.
    - Accordingly training the whole model from scratch is dropped.

- **Attempt 3** : 

    - Reference notebook : [S15_ControlNet_Inference_V1.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/Controlnet-canny/S15_ControlNet_Inference_V1.ipynb)
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

        ![a3-inference](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/d44ea153-e4dd-404c-82c3-020b2457ade3)

- **Attempt 4** : 

    - Reference notebook : [S15_ControlNet_V0.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/Controlnet-canny/S15_ControlNet_V0.ipynb)    
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
            ![a4-sample-created](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/31b5ad12-9f1c-4a9e-8251-76ca1b14d91a)
        - Reconstructed image
            ![a4-reconstructed-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/efec5013-5172-4943-acb1-efd8728b3aac)
        - Control image given as input for conditioning
            ![a4-control-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/8717ec71-a861-47e6-903d-90956d9d17d4)
        - Captions supplied
            ![a4-captions](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/ebd29660-875e-477e-8ddc-f5324756a3c3) 
    - **Conclusion** : This proves that pretrained model used further for training CTLNET with CDS provide good results compared to training the same from scratch.

- **Attempt 5** : 

    - Reference notebook : [S15_ControlNet_V4.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/Controlnet-canny/S15_ControlNet_V4.ipynb)  
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
            ![a5-sample-created](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/fd0d87ba-6c0f-4ef4-9290-260a7a6e9625)
        - Reconstructed image
            ![a5-reconstructed-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/4b74ee1b-ad6a-40f1-ae6d-d78caa573264)
        - Control image given as input for conditioning
            ![a5-control-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/777fda8e-6e1d-428b-9a72-4aa3cfc6ffcf)
        - Captions supplied
            ![a5-captions](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/99b097ca-18b9-4d1a-b9b5-9fda8245c8a3)
    - Inferencing also earned good results
       ![a5-inference](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/3223a136-4ccc-4b58-8cf5-2f4fe4ab855c)
    - **Conclusion** : This proves that pretrained model with trainable copy initialized and trained using an established dataset can provide good results conforming to the conditions set via control image and prompt.

- **Attempt 6** :

    - Reference notebook : [S15_ControlNet_V5.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/Controlnet-canny/S15_ControlNet_V5.ipynb)  
    - Preloaded the model with v1-5-pruned.ckpt (no canny edge involved so far).
    - Canny inferred a sample CDS image with this pretrained model before initializing the control-model weights.
    - Results were not resembling the canny edge captured.

        ![a6-inference-1](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/8e43ea8f-1012-4545-85d3-f0c51e4736e4)
    - Then, initialized the control-model weights (no canny edge involved again so far).
    - Again Canny inferred with the same sample CDS image.
    - Results were again not resembling the canny edge captured.

        ![a6-inference-2](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/114d5a76-6c82-4f5d-a649-e0dcb175317b)

        ![a6-inference-3](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/b28651b5-98bf-41d5-a8f9-b67d871c3f11)
    - These were done to ensure that pretrained model (without any canny training) as such cannot provide the results conforming to the conditions (in this case canny edges).
    - **Conclusion** : Results confirm the understanding i.e. model gave output image based on the prompt but output didn't conform to the canny edge control input provided. Hence fine-tuning for specific condition (in this case - canny) is necessary.

- **Attempt 7** : 
    - Reference notebook : [S15_ControlNet_V6.ipynb ](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/Controlnet-canny/S15_ControlNet_V6.ipynb) 
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

            ![a7-sample-created](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/539e9199-e001-4567-b8a9-1e5bf5babf00)
        - Reconstructed image

            ![a7-reconstructed-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/fa46a625-1529-4c58-9f26-a28dcb3e1245)
        - Control image given as input for conditioning

            ![a7-control-img](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/a41f4cfb-1233-435c-a7c1-2eff49d35ea4)
        - Captions supplied

            ![a7-captions](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/7f0c42e7-e53a-415c-9ff3-6298e19bb697)  
    - Inferred with the same model & results are promising

        ![a7-inference-1](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/c40a90f6-c296-40bf-940a-eafe3a970c18)

        ![a7-inference-2](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/4907326e-b5d7-449b-9875-4910a03f9e7f)

        ![a7-inference-3](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/b8937fbb-2b68-4cb8-8bd3-15453da19627)

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

