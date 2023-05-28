<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]

## Image Inpainting with Stable Diffusion Pipelines
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

- https://github.com/runwayml/stable-diffusion -> runwayml github repo having SD with inpainting details
- https://github.com/Stability-AI/stablediffusion/blob/main/scripts/gradio/inpainting.py -> Script for inpainting inference based on Gradio
- https://github.com/Stability-AI/stablediffusion -> StabilityAI github repo having SD with inpainting details
- https://github.com/anilbhatt1/stablediffusion -> Cloned repo of above with few edits to enbale running in colab
- https://huggingface.co/runwayml/stable-diffusion-inpainting -> Details on StableDiffusionInpaintPipeline
- https://github.com/huggingface/diffusers/tree/main/examples/research_projects/dreambooth_inpaint -> Training script for Dreambooth Inpaint training
- https://huggingface.co/docs/diffusers/main/en/using-diffusers/write_own_pipeline -> HuggingFace details on pipelines, models and schedulers

## Acronyms

* SD - Stable Diffusion
* HF - HuggingFace
* MI - Masked Image
* BM - Binary Mask
* OI - Original Image
* SDIP - StableDiffusionInpaintPipeline
* CDS - Custom DataSet

## Overview

- Stable Diffusion Inpainting is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input, with the extra capability of inpainting the pictures by using a mask.

![Inpaint-reconstruction](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/e6e53dd4-d54d-41c3-b081-d32dbc8b1a52)

- Stable Diffusion Inpainting high-level flow is as follows:

![Inpaint-Flow](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/f40032de-fa1a-4785-a3b7-57c1650f7149)

    - Original Image of size [3, 512, 512] is passed to VAE encoder which will convert it to latents of shape [4, 64, 64] 
    - Binary mask will be created. Part of the image where black patch is present will be white and rest of the mask will be black. 
    - Using this binary mask(BM) and original image(OI), a masked image (original image with black patch) will be created. 
    - This will be passed to VAE encoder which will convert it to latents of shape [4, 64, 64].
    - BM will be reshaped using torch.nn.interpolate to [1, 64, 64]
    - All these 3 latents will be combined using torch.cat to get a shape of [9, 64, 64]
    - This latent will be passed to UNET.
    - UNET will give an output latent of shape [4, 64, 64]
    - This will be passed to VAE decoder to eventually get output image with :
        - Either black patch removed (image reconstruction - no text prompt)
        - Or black patch replaced based on the text prompt

## Approach

- First, we will use established StableDiffusionInpaintPipeline(SDIP) from HF in google colab and see how inpainting (inference only) works against pretrained weights.
- Then we will train a UNET in smithsonian butterflies dataset and then with a CDS having flying objects.
- Based on this, we will check how the results look like when we incorporate our trained UNET in SDIP vs pretrained SDIP.
- Next, we will try another inference pipeline using Gradio and see how it works. This one also uses pretrained weights in pipeline.
- We will then understand how a SD inference pipeline is constructed for generating an image with a prompt only (no inpainting involved)
- Based on this understanding, we will built a custom pipeline for inpaint inferencing using pretrained weights from HF SD Inpaint models and see how results stack up. 

## Attempts-and-Results

 **Attempt 1** :

- We will use established StableDiffusionInpaintPipeline(SDIP) from HF in google colab and see how inpainting (inference only) works against pretrained weights.
- Notebook : [Link to S15_Inpainting_V1.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/StableDiffusion-Inpainting/S15_Inpainting_V1.ipynb)
- Results are good and image is reconstructed well

    ![a1-inference](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/90212fec-696b-4fc3-87cc-6fc425d5963a)

**Attempt 2** :

- Notebook : [Copy link for S15_Inpaint_Unet_V2.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/StableDiffusion-Inpainting/S15_Inpaint_Unet_V2.ipynb)
- Next we will train a UNET in smithsonian butterflies dataset.
    ![a2-butterfly-input](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/7fec5053-7d81-4a4c-9de4-9f6ce79dc95b)

- Training was done for 95 epochs for image size (3, 128, 128), batchsize - 16 against A100 GPU.
- Input given to UNET is as follows:
    - [B, 9, 16, 16] -> Latent input
        - [B, 4, 16, 16] -> OI latent (Converted from [3, 128, 128] via vae encoder)
        - [B, 4, 16, 16] -> MI latent (Converted from [3, 128, 128] via vae encoder)
        - [B, 1, 16, 16] -> BM latent (Converted via torch.nn.interpolate)
- Results were not that good.
    
    ![a2-training-result](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/03363a64-8a9a-40ca-9ed6-9fccdfc24032)

- Based on this training, we checked how inference results look like when we incorporate our trained UNET in SDIP vs pretrained SDIP.
- Results were not that great as expected for inference when SDIP was loaded with trained UNET.

    ![a2-inference-unet-trained-1](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/1dca9ea1-26a9-46cf-b122-0a9f808470aa)

- Results were obviously great when preatrained SDIP was used

    ![a2-inference-pretrained](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/6186630b-e18b-490a-a86f-4d1efcba0b69)

- Next, we tried another inference pipeline. 
- This one also uses pretrained weights in pipeline and accepts MI and BM as input.
- Check **Inferencing** section in the above notebook for code
- Reference script : https://github.com/Stability-AI/stablediffusion/blob/main/scripts/gradio/inpainting.py
- Results were decent for this inferencing as well.
    - Input 
    
        ![a2-gradio-input](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/fa4f5c18-716d-4a10-8d0a-7486aee83aba)

    - Inferred Result
    
        ![a2-gradio-inference](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/85534a8a-04d7-4cea-bc20-adf9ddb3c3b5)

- **Conclusion** : Training UNET from scratch and using it in SDIP may not be feasible.

**Attempt 3** :

- Notebook : [Copy link for S15_Inpaint_Unet_V3.ipynb](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/StableDiffusion-Inpainting/S15_Inpaint_Unet_V3.ipynb)
- Inorder to see if adding more variety of input data will help UNET perform better, we will train the UNET (trained on butterflies) with CDS also.
- CDS comprised of 10_397 images of Flying Birds and Small QuadCopters.

    ![a3-input](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/4aeac68d-3973-4fa4-908e-4406d378a587)

- Training was done for 1 epoch for image size (3, 128, 128), batchsize - 16 against A100 GPU.
- Results were not great as expected.

    ![a3-training-result-1](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/27076e1e-d9d0-494f-8887-ef0b483dc940)
    ![a3-training-result-2](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/48d2b933-fa62-4135-b6c7-19e402d3d8b5)

- **Conclusion** : Training UNET from scratch and using it in SDIP wont be feasible.

**Attempt 4** :

- We will now attempt to build a SD pipeline that will use pretrained weights and give inpainting results.
- To gain understanding on pipelines, we first used an SD inference pipeline that generates an image with a prompt only (no inpainting involved).
- Notebook : [Copy link for S15_Inpaint_Inference_V0](https://github.com/anilbhatt1/Deeplearning-E8P1/blob/master/StableDiffusion-Inpainting/S15_Inpaint_Inference_V0.ipynb)
- HF Reference for writing a pipeline : https://huggingface.co/docs/diffusers/main/en/using-diffusers/write_own_pipeline

- Then we created an SDIP with pretrained weights
- Notebook : Copy link for S15_Inpaint_Inference_V1
- Objective of inpainting is not to generate an image from scratch.
- Instead it deals with either reconstructing or planting something new in the mask location.
- Hence, the level of noise we need to start inferencing need not be 100%.
- We can control the noise we add by controlling the timesteps as shown below:
    ```
    scheduler = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")
    scheduler.config.num_train_timesteps -> 1000

    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)
        ->
        tensor([999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739,
        719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 500, 480, 460,
        440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180,
        160, 140, 120, 100,  80,  60,  40,  20])

    # Will execute the loop only for ([999, 979, 959, 939, 919, 899, 879, 859, 839, 819])
    infer_steps = 10
    for t in tqdm(scheduler.timesteps[:infer_steps]):
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t.to(dtype=torch.float32),         
                                encoder_hidden_states=text_embeddings).sample
    ```
- Accordingly various control levels and various latent combinations were tried out to see which one gave the best result.

**Attempt 4-A**
    - infer_steps = 10 
    
    - latent passed for noise reduction - OI + MI + BM -> [9, 64, 64] reshaped to [4, 64, 64] via 1x1 convolution
    
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    
    - Result
    
        ![a4a](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/9a006eb7-95f4-4ea7-aa79-d6b7dfb17518)        

**Attempt 4-B**
    - infer_steps = 30 
    
    - latent passed for noise reduction - OI + MI + BM -> [9, 64, 64] reshaped to [4, 64, 64] via 1x1 convolution
    
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    
    - Result
    
        ![a4b](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/e7c44c19-14f0-4f3a-bc39-4ee84ce4cc60)  

**Attempt 4-C**
    - infer_steps = 10 
    
    - latent passed for noise reduction - MI + BM -> [5, 64, 64] reshaped to [4, 64, 64] via 1x1 convolution
    
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    
    - Result
        
        ![a4c](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/f63f966a-ce97-48af-a3ab-20c10f38cada)
    
**Attempt 4-D**
    - infer_steps = 30 
    
    - latent passed for noise reduction - MI + BM -> [5, 64, 64] reshaped to [4, 64, 64] via 1x1 convolution
    
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    
    - Result
    
        ![a4d](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/68c25d7b-37dc-4faf-919a-33b7486b3ca6)

**Attempt 4-E**
    - infer_steps = 10 
    
    - latent passed for noise reduction - OI + BM -> [5, 64, 64] reshaped to [4, 64, 64] via 1x1 convolution
    
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    
    - Result
    
       ![a4e](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/32cc71be-a7a2-4984-9a9f-93bd58a0d178) 

**Attempt 4-F**
    - infer_steps = 30 
    
    - latent passed for noise reduction - OI + BM -> [5, 64, 64] reshaped to [4, 64, 64] via 1x1 convolution
    
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    
    - Result
    
        ![a4f](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/3cfe876a-c657-430b-8f90-81c4444ed28e)

**Attempt 4-G**
    - infer_steps = 50 
    
    - latent passed for noise reduction - MI-> [4, 64, 64] 
    
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    
    - Result

        ![a4g](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/a805503e-8d0c-464d-9f7a-0187086314c5)   
        
**Attempt 4-H**
    - infer_steps = 30 
    
    - latent passed for noise reduction - MI-> [4, 64, 64] 
    
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    
    - Result
    
        ![a4h](https://github.com/anilbhatt1/Deeplearning-E8P1/assets/43835604/7872b72f-7ace-44b5-8e5d-46803faeeb17)   
    
**Attempt 4-I**
    - infer_steps = 10 
    
    - latent passed for noise reduction - MI-> [4, 64, 64] 
    
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    
    - Result
    
   ![Uploading a4i.png…]()
     

**Attempt 4-J**
    - infer_steps = 50 
    - latent passed for noise reduction - OI-> [4, 64, 64] 
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    - Result

**Attempt 4-K**
    - infer_steps = 30 
    - latent passed for noise reduction - OI-> [4, 64, 64] 
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    - Result

**Attempt 4-L**
    - infer_steps = 10 
    - latent passed for noise reduction - OI-> [4, 64, 64] 
    - latent passed for noise prediction to UNET - OI + MI + BM -> [9, 64, 64]
    - Result

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
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555

