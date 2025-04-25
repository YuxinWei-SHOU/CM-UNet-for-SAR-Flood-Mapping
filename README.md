# Satellite Data and AI Reveal Rapidly Growing Food Vulnerability in Bangladesh

## Overview

We propose a CNN-Mmaba-UNet (CM-UNet) model for **Bitemporal and Dual-Polarization SAR flood mapping**, aimed at revealing rapidly growing food vulnerability in Bangladesh. This model can be used for dynamic flood analysis and uncovering socioeconomic vulnerability at the national scale.

## Available Datasets

### 1. Dual-Polarization Synthetic Aperture Radar (SAR) for Water Body Segmentation
This dataset includes SAR imagery collected in Bangladesh from 2015 to 2024, specifically designed for water body segmentation using Dual-Polarization SAR.  
[Download Dataset](https://zenodo.org/records/15255691)

### 2. Bitemporal and Dual-Polarization SAR for Dynamic Flood Mapping and Socioeconomic Analysis
This dataset provides 4-band SAR data, national-scale flood binary maps, and socioeconomic data (including land use, night-time light data, and population statistics) collected from 2015 to 2024.  
[Download Dataset](https://zenodo.org/records/15255271)

## Figures

### Figure 1: The Architecture of CM-UNet Model
The architecture of the **Convolutional Neural Network and Mamba combined UNet model (CM-UNet)**, designed for processing SAR imagery. The model uses both **VV** (Vertical transmission and Vertical reception) and **VH** (Vertical transmission and Horizontal reception) polarizations. Pre-T and Post-T represent the pre- and post-flood events, respectively. The model incorporates a **2D Selective Scan Module (SS2D)**, **Visual State-Space (VSS)**, and **Selective Scan Space State Sequential Model (S6)**. The tensor size in CM-UNet is represented as $\frac{H}{L}\times\frac{W}{L}\times C$, with $L$ taking values of 2, 4, and 8, and $C$ representing the channels (128, 256, 512).

![Figure 1](https://github.com/YuxinWei-SHOU/CM-UNet-for-SAR-Flood-Mapping/blob/c01417007421b2b3b5d7ddfb586325c4ea3c1c31/assets/Figure_1.jpg)

### Figure 2: Workflow of the Multi-temporal Trend Analysis Process for National-Scale Flood Mapping
This figure illustrates the workflow of the **multi-temporal trend analysis** process for mapping national-scale flood events.

![Figure 2](https://github.com/YuxinWei-SHOU/CM-UNet-for-SAR-Flood-Mapping/blob/c01417007421b2b3b5d7ddfb586325c4ea3c1c31/assets/Figure_2.jpg)

### Figure 3: Spatial Distribution of Flood Risk and Population Dynamics in Bangladesh
This figure presents spatial distribution of flood risk and population dynamics in Bangladesh.

![Figure 3](https://github.com/YuxinWei-SHOU/CM-UNet-for-SAR-Flood-Mapping/blob/c01417007421b2b3b5d7ddfb586325c4ea3c1c31/assets/Figure_3.jpg)

### ***Note:***  
**The project is currently in the submission phase, and further updates will be made once the paper is published. Stay tuned for the latest information!**
