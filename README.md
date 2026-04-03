# Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer (ICLR 2026)
<table>
  <tr>
    <td> <img src = "figures/figure1.jpg"> </td>
  </tr>
</table>


## :bulb: Highlight
 - :heart_eyes: :heart_eyes: Color3D is a unified controllable 3D colorization framework for both static and dynamic scenes, producing vivid and chromatically rich renderings with strong cross-view and cross-time consistency.

## :label: TODO 
- [x] Release video demo.
- [x] Release codes for personalized colorizer.
- [x] Release training codes.

## :medal_military: Framework Architecture
<table>
  <tr>
    <td> <img src = "figures/figure2.jpg"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of Color3D</b></p></td>
  </tr>
</table>

   

## 🎨 Automatic Colorization Pipeline

### 🛠️ 1. Environment Setup

Before running the scripts, ensure your environment meets the requirements (e.g., 3DGS + requirements.txt).


### 📥 2. Pretrained Model Preparation
You need the ddcolor_paper_tiny.pth weights to ./stage1/.

Official Download: https://github.com/piddnad/DDColor/tree/master

### 🚀 3. Usage Guide
Key view colorization
```
python key_view_colorization.py --folder_path ./example/gray
```

Stage 1: Personalized Colorizer Training

```
cd stage1
```

Start Stage 1 training:

```
python train.py
```

Generate intermediate images:

```
python generate_images.py
```

Stage 2: 3D Colorization

```
cd ../stage2
```

Start Stage 2 training:

```
python train.py -s ../example
```
