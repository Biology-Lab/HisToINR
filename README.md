# HisToINR: Integrating Histological Images with Implicit Neural Representations for Spatial Transcriptomics Analysis
HisToINR is a unified framework for spatial transcriptomics analysis that integrates transcriptomic profiles, histological image information, spatial coordinates, and cross-slice context through implicit neural representations (INRs) and residual multimodal fusion. 

The framework supports multiple downstream tasks, including:

- Spatial Domain Identification
- Cell-type Deconvolution
- Gene Denoising
- Multi-omics Integration

# Environment Setup
| Component | Version         |
| --------- | --------------- |
| Python    | 3.12            |
| CUDA      | 12.8            |
| PyTorch   | 2.7.0           |

# Framework
![Framework](framework.png)

The overall workflow of HisToINR consists of four major components:

### 1. INR-based Spatial Representation Learning
Spatial coordinates $(x,y,z)$ are encoded using sinusoidal implicit neural representations (INRs) to reconstruct gene expression profiles.
### 2. Histological Image Feature Extraction
Histological image patches are extracted around each spot and encoded using the pretrained UNI foundation model.
### 3. Morphology-aware Residual Fusion
Transcriptomic latent representations and morphological features are integrated using a residual fusion strategy:

$$Z_{fusion} = Z_{spatial} + \alpha Z_{img}$$

where $\alpha$ is a learnable fusion coefficient.

---
### 4. Downstream Analysis
The fused representations support:
- Spatial Domain Identification
- Gene Denoising
- Cell-type Deconvolution
