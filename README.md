# Stream Segment Anything

This project processes **.mov** dashcam video files from a specified directory, extracts frames, and applies Facebook’s **Segment Anything Model (SAM)** to each. It then generates segmentation masks, displaying both the original frame and the generated mask in real-time.

Super slow when running on CPU. Needs GPU for closer to real time segmentation. This folder processes one jpg at a time in slow offline/batch mode.


### WARNING: Have retained the streaming script for now, but really the batch mode is the only thing tested so far.
---

## Installation

To get started, you'll need **uv**. If you don't have it yet, you can install it by following the instructions on the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

Once **uv** is installed, navigate to your project's root directory and run the following command to install all the necessary dependencies:

Bash

```
uv sync
```

---

## Model Setup

To use SAM, you'll need to download a pre-trained model checkpoint. We recommend the **ViT-H checkpoint**.

📥 **Download here:** 
Large model: [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_4b8939.pth)
Smaller base model: [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

Place the downloaded **.pth** file in your project's root directory. The script is configured to look for this file by default.

---

## Usage

Before running the script, make sure your video files are located in the `bdd100k/videos/train/` directory, relative to your project's root. The script is specifically configured to look for **.mov** files within this path.

If you need some sample videos, you can find a 50-video subset from the large bdd100k dataset here: [bdd100k 50 video sample from Kaggle](https://www.kaggle.com/datasets/deeplyft/driving-video-subset-50-with-object-tracking)

To execute the project, use `uv run`:

First use the frame extractor to reorient and convert a video into a folder of jpgs:

```
uv run extract_frames.py
```

The script below will run through the images one at a time and auto generate a mask and overlay it on the original image. Very very slowly!

Bash

```
uv run check_sam_mask.py
```

---

## Key Library Dependencies

- **OpenCV (`opencv-python`)**: Handles video capture, frame extraction, and basic image manipulations like color conversion and rotation.
    - [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- **Segment Anything (`segment-anything`)**: The core library for applying the Segment Anything Model.
    - [Segment Anything GitHub Repository](https://github.com/facebookresearch/segment-anything)
- **Pillow (`Pillow`)**: Used for image handling, specifically converting NumPy arrays to PIL Image objects, which may be required for certain operations or visualizations.
    - [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)
- **NumPy (`numpy`)**: Provides essential numerical operations, especially for managing image data as arrays.
    - [NumPy Documentation](https://numpy.org/doc/stable/)
- **PyTorch (`torch`)**: The deep learning framework that powers the Segment Anything Model for execution.
    - [PyTorch Documentation](https://pytorch.org/docs/stable/)
- **Torchvision (`torchvision`)**: A PyTorch companion library that provides datasets, models, and image transformations.
    - [Torchvision Documentation](https://pytorch.org/vision/stable/index.html)