# AML 2021/2022 "Real-time Domain Adaptation in Semantic Segmentation" Project
This repository includes the full code and the paper of the project for the final exam of "Advanced Machine Learning" course @ Politecnico di Torino.  

## Code

The code contains:
1. An implementation of a real-time semantic segmentation network, BiSeNet (Bilateral Segmentation Network), that can exploit two different backbones, ResNet-101 or ResNet-18;
2. An implementation of an unsupervised adversarial domain adaptation algorithm;
3. A variation of the unsupervised adversarial domain adaptation algorithm with lightweight depthwise-separable convolutions for the adversarial discriminator, which significantly reduce the total number of parameters and the total number of Floating Point Operations (FLOPS) of the model, making it suitable for mobile and/or embedded devices;
4. Two image-to-image transformations, to improve domain adaptation:
    * FDA;
    * LAB;
5. Generation of pseudo labels for the target domain, to further enhance domain adaptation;
6. A combination of LAB transformation and pseudo labels generation for the target domain.  

## Datasets
* [Subsets of Cityscapes and GTA5](https://mega.nz/file/ERkiQBaY#h-wktK7U7MpIG5nf-rMWF7d76NEM5ae_MrAmELftNR0)

## Notebooks
The notebook "Real-time Domain Adaptation in Semantic Segmentation.ipynb" includes all the results, except for the last one that is in the "Real-time Domain Adaptation in Semantic Segmentation (2).ipynb" notebook.