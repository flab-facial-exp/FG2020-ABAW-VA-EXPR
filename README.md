# Multi-term and Multi-task Affect Analysis in the Wild 

Challenges: **FG-2020 Competition: Affective Behavior Analysis in-the-wild (ABAW)**

URL: https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/

Team Name: **FLAB2020**

Team Members: Sachihiro Youoku,  Junya Saito, Yuushi Toyoda, Ryosuke Kawamura, Takahisa Yamamoto, Xiaoyu Mi, Kentaro Murase

Affiliation: Trusted AI Project, Artificial Intelligence Laboratory, Fujitsu Laboratories Ltd., Japan

The paper link: [Multi-term \& Multi-task Affect Analysis in the Wild](arxiv)

## Update:

- 2020.10.XX: release

## How to run

 We use opensource library [*Openface*](https://github.com/TadasBaltrusaitis/OpenFace), [*Openpose*](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for generating features

1. Download and setup Anaconda3

2. Install dependencies

   ```
   pip install lightgbm
   pip install opencv-python
   ```

3. Download Dataset

   - We use [Aff-Wild2 database](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)

4. Download & setup Openface
   - Download [Openface](https://github.com/TadasBaltrusaitis/OpenFace)
   - Generate Openface features (AU, pose, gaze) from videos in Aff-Wild2 database
   - Copy generated csv files to the directory 'base_data/OpenFace'

5. Download & setup Openpose
   - Dowload [Opnpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
   - Generate Openpose features from videos in Aff-Wild2 database
   - Convert generated json files to '(video name)_openpose.csv' files
   - Copy csv files to  the directory 'base_data/OpenPose'

6. setup ResNet50
   - Generate ResNet50 features using ['*TORCHVISION.MODELS*'](https://pytorch.org/docs/stable/torchvision/models.html) from cropped image in Aff-Wild2 database.

   - using pre-trained model, below:

     ```
     import torchvision.models as models
     resnet50 = models.resnet50(pretrained=True)
     ```

   - Convert ResNet50 features to '(video name)_resnet50.h5' files
     
     - note: Dimensional reduction to 200 dimensions using PCA
     
   - Copy csv files to  the directory 'base_data/Resnet'

7. setup EfficientNet

   - Generate EfficientNet features using [GitHub Model](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
   - Convert EfficientNet features to '(video name)_enet.h5' files
     - note: Dimensional reduction to 300 dimensions using PCA
   - Copy csv files to  the directory 'base_data/Enet'

8. Run terminal
   ```
   cd ~(this directory)
   jupyter lab
   ```

9. Run Submit X 
   - Execute 00~ to 10~ ipynb files in order
   - 00: calculate frame count, and save per videos
   - 01: merge features (openface, openpose, ...) and label data
   - 02: create multi-term features data from merged data
   - 03: generate single-term models
   - 04: generate single-task models using single-term models
   - 05: generate multi-task models using single-task models and single-term models
   - 06: create multi-term features data for validation per frame
   - 07: predict and evaluate validation dataset per frame
   - 08: merge features (openface, openpose, ...) for test dataset
   - 09: create multi-term features data for  test dataset per frame
   - 10: predict test dataset

## Framework

- Overview:
- Preprocessing:
- Multi-term \& Multi-task Model:
- 

## Citation

```
@misc{flab2020affect,
    title={Multi-term and Multi-task Affect Analysis in the Wild },
    author={Sachihiro Youoku and Yuushi Toyoda and Ryosuke Kawamura and Junya Saito and Takahisa Yamamoto and Xiaoyu Mi and Kentaro Murase},
    year={2020},
    eprint={2002.03399},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
