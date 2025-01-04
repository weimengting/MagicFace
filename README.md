# MagicFace
Official implementation of MagicFace

> **MagicFace: High-Fidelity Facial Expression Editing with Action-Unit Control** [[arXiv paper]()]<br>
> Mengting Wei, Tuomas Varanka, Xingxun Jiang, Huai-Qian Khor, Guoying Zhao<br>
> University of Oulu


## Introduction
We address the problem of facial expression editing
by controling the relative variation of facial action-unit (AU) from
the same person. This enables us to edit this specific person’s expression in a fine-grained, continuous and interpretable manner,
while preserving their identity, pose, background and detailed
facial attributes. By injecting AU variations
into a denoising UNet, our model can animate arbitrary identities
with various AU combinations, yielding superior results in high-fidelity expression editing compared to other facial expression
editing works.

<p align="center">
  <img src="./assets/demo.jpg" />
</p>

### Dependencies

- Python 3.10
- Your computer should have a graphics card with approximately **24GB** to support running this test.

### Installation

You can first create a new Python 3.10 environment using `conda` and then install this package using `pip` from the PyPI hub:

```console
conda create -n magicface python=3.10
conda activate magicface
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Usage

#### Using our examples

You can test our model by editing the images we provided. Model inference needs an identity image
to edit, a background image for attribute condition and an AU condition. 

1. First, download the pre-trained weights of the denoising UNet from [OneDrive](https://unioulu-my.sharepoint.com/:u:/g/personal/mwei23_univ_yo_oulu_fi/Ee674DGMLF1Lsh_UZtPOaBgBlIfYuylsxtF_vsSRGjHxzQ?e=1t6quP)
and that of the ID encoder from [OneDrive](https://unioulu-my.sharepoint.com/:u:/g/personal/mwei23_univ_yo_oulu_fi/ETpI7WscliBAj7KK7qFlXvcBmS3f4qn3RCIYU6oj7-erpg?e=FnPLrS).

```console
mkdir pre_trained
cd pre_trained
mkdir denoising_unet
mkdir ID_enc
```

Place the downloaded files in the appropriate directory and then unzip them. The structure of
the directory for pre_trained should be like

```commandline
pre_trained/
   denoising_unet/
       checkpoint-100000/ 
            diffusion_pytorch_model.safetensors
            config.json
   ID_enc/
       checkpoint-100000/ 
            diffusion_pytorch_model.safetensors
            config.json
```

2. Test the model:

```--au_test``` The AUs you want to modify for the face. We provide 12 editable AUs here.
They are _AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26_. Only provide the AUs 
you want to modify here and split them by ``+``. For example, the following example shows how to 
edit AU1 and AU4. If only one AU is intended to change, just provide that one.

```--AU_variation```  Intensity integers you want to edit for each AU you specified. Also split them by ``+`` 
if multiple AUs are intended to change. We recommend to limit the intensity 
in the range of [-8, 8]. Integers outside this range may experience severe distortion.


```console
python inference.py --img_path './test_images/00381.png' --bg_path './test_images/00381_bg.png' --au_test '4+1' --AU_variation '4+2'
```

#### Test your own images

If you want to edit your own images, you need to compute the 
background and pose for attribute condition.

1. Download the utils that will be used for cropping and parsing the face from [OneDrive](https://unioulu-my.sharepoint.com/:u:/g/personal/mwei23_univ_yo_oulu_fi/EVASuyMAoSJKrEqiTmwouKEB65bb4xAjuKkVervBrXNbHA?e=HoGghH).
Unzip it, take out the files and put them under the `utils` directory. 
2. Crop your image into the resolution of 512 $\times$ 512. Please provide
the image including more than one face, otherwise it will result in an error.
```console
cd utils
python preprocess.py --img_path <your-image-path> --save_path <your-save-path>
```

3. Then parse the background and draw the contour from the cropped image.

```console
python retrieve_bg.py --img_path <your-cropped-path> --save_path <your-save-path>
```
4. Use the inference.py script introduced above to test your image.
### Issues or Questions?
If the issue is code-related, please open an issue here.

For questions, please also consider opening an issue as it may benefit future reader. 
Otherwise, email Mengting Wei at [mengting.wei@oulu.fi](mengting.wei@oulu.fi).

### Acknowledgements

This codebase was built upon and drew inspirations from [FineFace](https://github.com/tvaranka/fineface), [InsightFace](https://github.com/deepinsight/insightface),
[BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) and [Stable DIffusion](https://github.com/CompVis/stable-diffusion). 

We thank the authors for making those repositories public.