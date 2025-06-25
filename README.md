# Stream Depth Anything

This project processes video files (`.mov`) from a specified directory, extracts frames, and applies a depth estimation model to each. It uses the **`depth-anything/Depth-Anything-V2-Small-hf`** model from Hugging Face Transformers to generate depth maps, displaying both the original frame and the depth map in real-time.

---

## Installation

To get started, you'll need **`uv`**. If you don't have it yet, you can install it by following the instructions on the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

Once `uv` is installed, navigate to your project's root directory and run the following command to install all the necessary dependencies:

Bash

```
uv sync
```

---
## Usage

Before running the script, make sure your video files are located in the `bdd100k/videos/train/` directory, relative to your project's root. The script is specifically configured to look for **`.mov`** files within this path.

If needed, you can find a 50 video sample from the large bdd100k dataset here: [bdd100k 50 video sample from Kaggle](https://www.kaggle.com/datasets/deeplyft/driving-video-subset-50-with-object-tracking)

To execute the project, use `uv run`:

Bash

```
uv run stream_depth_anything.py
```

The script will go through each `.mov` file found in the specified directory, process it frame by frame, and display the original frame alongside its corresponding depth map. You can press the **`q`** key at any time to quit the display for the current video and move to the next one.

---
## Key Library Dependencies
- **OpenCV (`opencv-python`)**: Handles video capture, frame extraction, and basic image manipulations like color conversion and rotation.
    - [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- **Hugging Face Transformers (`transformers`)**: Crucial for loading and using the pre-trained depth estimation model (**`depth-anything/Depth-Anything-V2-Small-hf`**).
    - [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- **Pillow (`Pillow`)**: Used for image handling, specifically converting NumPy arrays to PIL Image objects, which are required as input for the Hugging Face pipeline.
    - [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)
- **NumPy (`numpy`)**: Provides essential numerical operations, especially for managing image data as arrays.
    - [NumPy Documentation](https://numpy.org/doc/stable/)
- **PyTorch (`torch`)**: The deep learning framework that powers the Hugging Face Transformers library for model execution.
    - [PyTorch Documentation](https://pytorch.org/docs/stable/)