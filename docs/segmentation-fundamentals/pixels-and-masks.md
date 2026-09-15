# Pixels and Masks

## Overview

Before diving into segmentation architectures and training pipelines, it's essential to understand the fundamental building blocks of every segmentation task: pixels and masks.

Every segmentation model — no matter how sophisticated — ultimately operates on these two concepts:

- Pixels are the raw units of image data — the input.
- Masks are the encoded outputs — the segmentation labels.

Understanding how pixels are represented, how masks are constructed, and how they relate to each other is foundational to everything that follows.

---

## What is a Pixel?

A pixel (short for "picture element") is the smallest addressable unit of a digital image. Each pixel holds a numerical value (or set of values) that represents color or intensity at a specific spatial location.

### Pixel Representation by Image Type

| Image Type              | Channels               | Value Range               | Example                     |
|-------------------------|------------------------|---------------------------|-----------------------------|
| Grayscale               | 1                      | 0–255 (uint8) or 0–1 (float) | X-ray, CT slice             |
| RGB                     | 3 (R, G, B)           | 0–255 per channel         | Natural photos              |
| RGBA                    | 4 (R, G, B, Alpha)    | 0–255 per channel         | Images with transparency     |
| Multi-channel Medical    | N (e.g., 4 for BraTS MRI) | Varies                  | T1, T1ce, T2, FLAIR stacked |
| Hyperspectral            | 100+                   | Varies                    | Satellite, specialized medical |

### How Pixels Form an Image

An image is a 2D grid of pixels with a defined height (H) and width (W):

```
Image = H × W × C

Where:
  H = height (rows)
  W = width (columns)
  C = number of channels
```

For example:

- A grayscale CT slice: 512 × 512 × 1
- An RGB photo: 1920 × 1080 × 3
- A BraTS MRI volume: 240 × 240 × 155 × 4 (H × W × Depth × Modalities)

### Pixel Coordinates

Each pixel has a coordinate (x, y) or (row, col):

```
(0,0)  (0,1)  (0,2)  ...  (0,W-1)
(1,0)  (1,1)  (1,2)  ...  (1,W-1)
...
(H-1,0) ...              (H-1,W-1)
```

This grid structure is why Convolutional Neural Networks (CNNs) work so naturally on images — they exploit spatial locality.

---

## What is a Mask?

A mask is the output of a segmentation model — a spatially aligned image where each pixel's value encodes its class or region assignment.

Think of a mask as a label map: for every pixel in the input image, the mask tells you what that pixel is.

### Key Properties of a Mask

- Same spatial dimensions as the input image (H × W).
- Values represent classes, not colors (though we colorize them for visualization).
- Often single-channel even for multi-class problems (integer labels).
- Can be binary (0 or 1) or multi-class (0, 1, 2, ..., K).

### Types of Masks

1. **Binary Mask**

Two values only — typically 0 (background) and 1 (foreground).

```
Example: Tumor vs. Non-tumor
[[0, 0, 0, 0],
 [0, 1, 1, 0],
 [0, 1, 1, 0],
 [0, 0, 0, 0]]
```
Used in: binary segmentation tasks.

2. **Multi-class Mask**

Each pixel holds an integer class label.

```
Example: Brain tumor sub-regions
0 = background
1 = necrotic core
2 = edema
3 = enhancing tumor

[[0, 0, 0, 0],
 [0, 2, 2, 0],
 [0, 2, 3, 0],
 [0, 0, 0, 0]]
```
Used in: multi-class segmentation tasks.

3. **Multi-channel / One-hot Mask**

For deep learning, masks are often converted into one-hot encoded tensors:

```
Class 0 (bg):  [[1,1,1,1], [1,0,0,1], ...]
Class 1:       [[0,0,0,0], [0,1,1,0], ...]
Class 2:       [[0,0,0,0], [0,0,0,0], ...]
```

Shape becomes H × W × K (K = number of classes). This is required for softmax + cross-entropy loss.

4. **Instance Mask**

Each object gets a unique ID, even within the same class.

```
0 = background
1 = tumor #1
2 = tumor #2
3 = tumor #3
```
Used in: instance segmentation.

5. **Probability Mask / Soft Mask**

Instead of hard labels, each pixel holds a probability (0.0–1.0) of belonging to a class.

```
[[0.02, 0.05, 0.91, 0.03],
 [0.10, 0.88, 0.95, 0.04],
 ...]
```
Used during training; converted to hard masks via argmax at inference.

---

## Pixels → Masks: The Core Idea

Segmentation is fundamentally a pixel-wise classification problem:

```
For each pixel (x, y) in image I:
    mask(x, y) = class_of_pixel(I, x, y)
```

Unlike image classification (one label per image) or object detection (bounding boxes), segmentation produces one label per pixel.

### Visualizing the Transformation

```
Input Image (H × W × C)    →    Model    →    Mask (H × W)

┌─────────────────┐                          ┌─────────────────┐
│ ░░░▒▒▒▒░░░░░░░░ │                          │ 000111100000000 │
│ ░░▒▒███▒▒░░░░░░ │                          │ 000111222000000 │
│ ░▒▒██████▒░░░░░ │         →                │ 001112222100000 │
│ ░░░▒▒███▒▒░░░░░ │                          │ 000111222000000 │
│ ░░░░░▒▒▒░░░░░░░ │                          │ 000001110000000 │
└─────────────────┘                          └─────────────────┘
  Raw pixel values                            Class labels
```

The mask is spatially aligned with the input — pixel (x, y) in the mask corresponds exactly to pixel (x, y) in the image.

---

## How Masks Are Created (Ground Truth)

Ground truth masks are created through annotation:

1. **Manual Annotation**

   - Experts (radiologists, pathologists) draw boundaries pixel by pixel.
   - Tools: ITK-SNAP, 3D Slicer, LabelMe, CVAT, Supervisely.
   - Expensive and time-consuming — a single 3D MRI volume can take hours.

2. **Semi-Automated Annotation**

   - Model-assisted labeling (human corrects model output).
   - Reduces annotation time significantly.

3. **Weakly Supervised**

   - Derived from image-level labels, bounding boxes, or scribbles.
   - Cheaper but noisier.

4. **Synthetic / Simulated**

   - Generated from phantoms or simulations.
   - Useful for pretraining.

### The Annotation Quality Problem

Ground truth masks are not perfect:

- Different annotators produce different masks (inter-observer variability).
- The same annotator may produce different masks on different days (intra-observer variability).
- Boundaries in medical images are often inherently ambiguous.

This means segmentation models are learning from noisy labels — an important reality to keep in mind.

---

## Masks in Code: Practical Representation

### Python / NumPy

```python
import numpy as np

# Binary mask
binary_mask = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], dtype=np.uint8)

# Multi-class mask
multiclass_mask = np.array([[0, 0, 0],
                            [0, 1, 2],
                            [0, 0, 0]], dtype=np.uint8)

# One-hot encoded mask
num_classes = 3
one_hot = np.eye(num_classes)[multiclass_mask]  # shape: (3, 3, 3)
```

### PyTorch

```python
import torch

# Typical mask tensor shape: (Batch, Height, Width)
mask = torch.tensor([[0, 0, 0],
                     [0, 1, 2],
                     [0, 0, 0]], dtype=torch.long)

# For loss computation, often converted to (Batch, Classes, H, W)
one_hot = torch.nn.functional.one_hot(mask, num_classes=3)
one_hot = one_hot.permute(2, 0, 1).unsqueeze(0).float()
```

### TensorFlow / Keras

```python
import tensorflow as tf

mask = tf.constant([[0, 0, 0],
                    [0, 1, 2],
                    [0, 0, 0]], dtype=tf.int32)

one_hot = tf.one_hot(mask, depth=3)  # shape: (3, 3, 3)
```

### Common Shape Conventions

| Framework            | Image Shape        | Mask Shape                |
|----------------------|--------------------|---------------------------|
| PyTorch (2D)        | (B, C, H, W)       | (B, H, W) or (B, K, H, W) |
| PyTorch (3D)        | (B, C, D, H, W)    | (B, D, H, W) or (B, K, D, H, W) |
| TensorFlow           | (B, H, W, C)       | (B, H, W) or (B, H, W, K) |

Mismatched shape conventions are one of the most common sources of bugs in segmentation code.

---

## Visualizing Masks

Raw masks are just numbers — hard to interpret. We colorize them for visualization:

```python
import matplotlib.pyplot as plt
import numpy as np

# Define a color palette for classes
palette = {
    0: (0, 0, 0),        # background - black
    1: (255, 0, 0),      # class 1 - red
    2: (0, 255, 0),      # class 2 - green
    3: (0, 0, 255),      # class 3 - blue
}

def colorize_mask(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        rgb[mask == cls] = color
    return rgb

# Overlay on original image
def overlay(image, mask, alpha=0.5):
    colored = colorize_mask(mask)
    return (image * (1 - alpha) + colored * alpha).astype(np.uint8)
```

### Common Visualization Conventions in Medical Imaging

| Color           | Meaning                                     |
|------------------|---------------------------------------------|
| Red              | Enhancing tumor (ET)                        |
| Green            | Tumor core (TC)                            |
| Yellow/Blue      | Whole tumor (WT) / Edema                   |

Consistent color coding helps clinicians and researchers interpret results quickly.

---

## Masks and Loss Functions

The relationship between pixels and masks directly influences loss function design:

- **Cross-Entropy Loss:** Treats every pixel as an independent classification problem.
- **Dice Loss:** Measures overlap between predicted and ground-truth masks.
- **Focal Loss:** Down-weights easy pixels — critical when background dominates.
- **Boundary Loss:** Focuses on pixels near mask edges.

Masks are not just outputs — they are training signals. Their quality and structure directly shape what the model learns.

---

## Common Pitfalls

1. **Shape Mismatches:** Mask and image must have identical H × W. Resizing one without the other breaks alignment.

2. **Wrong Data Types:** Masks should be uint8 or int64 (for class labels) — not float. Floating-point masks cause silent bugs in loss computation.

3. **Ignoring Class Imbalance:** A mask with 99% background pixels will bias the model toward predicting background.

4. **Interpolation Errors:** When resizing masks, always use nearest-neighbor interpolation — never bilinear (which creates invalid labels).

5. **Color Palette Confusion:** Don't confuse visualization colors with actual mask values. A "red" region in a colored mask might have value 1, not (255, 0, 0).

6. **Channel Order:** PyTorch uses (C, H, W), TensorFlow uses (H, W, C). Mixing them silently corrupts training.

---

## A Mental Model: The Coloring Book Analogy

Think of segmentation like a coloring book:

- The input image is the photograph you're tracing.
- The pixels are the tiny dots you trace along.
- The mask is the traced outline filled with a specific color per region.
- The model is the artist learning to trace accurately.

Different color = different class. Multiple separate shapes of the same color = semantic segmentation. Unique colors per shape = instance segmentation.

---

## Summary

| Concept         | Role                      | Representation                      |
|------------------|---------------------------|-------------------------------------|
| Pixel            | Raw input unit            | Numeric value(s) at (x, y)        |
| Image            | Grid of pixels            | H × W × C tensor                   |
| Mask             | Segmentation output       | H × W label map                    |
| Binary Mask      | Two classes               | 0 or 1                             |
| Multi-class Mask | K classes                 | 0, 1, ..., K-1                     |
| One-hot Mask     | DL-friendly               | H × W × K                          |
| Instance Mask    | Unique per object         | 0, 1, 2, ..., N                    |
| Ground Truth     | Training target           | Expert annotation                   |

Pixels and masks are the atomic units of segmentation. Every architecture, loss function, and metric ultimately operates on them. Mastering their representation, manipulation, and visualization is the first step toward building robust segmentation systems.

In the next section, we'll see how these concepts come together in binary segmentation — the simplest and most common segmentation task.

---

## Further Reading

- Binary Segmentation →
- Multi-class Segmentation →
- Segmentation Pipeline →
- What is Image Segmentation?

---

Next: Binary Segmentation →

