
# Face recognition with/without mask based on Mask Learning with Pairwise Differential Siamese Network

This is the reproducible code using Pairwise Differential Siamese Network to apply for Face Recognition with mask based on the paper [https://arxiv.org/pdf/1908.06290.pdf].




## Introduction

Pairwise Differential Siamese Network has been one of the state-of-the-art models for Face Recognition with occlusions (mainly for masked faces) in recent years.

The master branch works with Google Colab using a local runtime consists of 32 GB of RAM and the NVIDIA GeForce RTX 3060 or a hosted runtime of A100 GPU with High-RAM.

The code serves for the purpose of performance and computation comparison between Face Recognition with mask capabilities of the original Siamese Neural Network, the famous FaceNet and the Pairwise Differential Siamese Network (PDSN) and might be used for future works relating to Face Recognition.


## Installation

### Datasets

The used LFW and MLFW datasets can be downloaded at: \
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset  
http://whdeng.cn/mlfw/?reload=true

### Data Preprocessing

The training, validation and testing datasets should be loaded and read using PIL image. PIL can be installed via pip.

```python
pip install Pillow
```

Since all images need to be at the same size, images from LFW datasets should be resized to match that of MLFW datasets using Pytorch transform.

```python
torchvision.transforms.Resize((128, 128))(input_image)
```

### Training, Validation and Testing 

Training Set (60% of datasets)

* Path: ```{Path to your training pairs list}/pairsDevTrain.txt```

* Sample structure

```David_George_0001.jpg   David_George_0002.jpg   1``` \
```James_Colbert_0001.jpg  Zinedine_Zidane_0001.jpg    0```

Validation Set (20% of remaining datasets)

* Path: ```{Path to your training pairs list}/pairsDevVal.txt```

Testing Set (20% of remaining datasets)

* Path: ```{Path to your training pairs list}/pairsDevTest.txt```

The ```default_reader_pair()``` function receives an input of the above datasets and outputs pairs of images with their correct labels then passed through the class ```ImageList_pair``` to loop for each pair while the ```default_reader()``` and the ```default_reader_pairmask()``` are used for loading a single image list with labels or generating a masked pair image list with labels respectively.

### Model

The Pairwise Differential Siamese Network is constructed using the class ```LResNet_Siamese()``` using PyTorch. 

### Classifier

There are multiple classifier options ranging from MCP, Angle Linear and Linear but if we want to choose MCP then we need to compile the class ```MarginCosineProduct()``` from the Layer. Otherwise we can contruct a fully connected layer.

### Utility

In Utils, there are a lot of utility functions for further usage, one of the most used functions is the ```accuracy()``` in which it offers a calculation of accuracy when training though each epoch. ```load_state_dict()``` is also widely used for loading weights for pre-trained models.

### Run model training

Before training the model, please input these parameters:

* ```train_list``` : Training datasets
* ```valid_list``` : Validation datasets
* ```batch_size``` : Batch size
* ```num_class``` : Number of people need to be recognized
* ```classifier_type``` : MCP, AL or L
* ```epochs``` : Number of epochs
* ```lr``` : Learning rate

Once all parameters have been input, run the ```main()``` function in ```main dictionary```.

### Post training

After training the model and saving weights, the model can be tested using the same code of the function ```validation()``` by loading the ```classifier``` with the Testing datasets.

```
label_list = label.tolist() #label is the list of all labels in the Testing datasets
valid_transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
])

test_loader = torch.utils.data.DataLoader(
    ImageList_pair(root='{Your path to Testing datasets}',
                  transform=valid_transform),
    batch_size=1, shuffle=False,
    num_workers=0, pin_memory=False, drop_last=False)

with torch.no_grad():
  for batch_idx, (data, data_occ, _, target) in enumerate(test_loader, 1):
      f_masked, focc_masked, output, output_occ, _, _ = model_test(data, data_occ)
      pred = output
      output = classifier(output, label)
      print(label_list.index(target), output.argmax(), target)
      if (output.argmax().item() == label_list.index(target)):
          i = i + 1

```
## Citation

Please consult and consider citing the following paper:

```@INPROCEEDINGS{9009826,
  author={Song, Lingxue and Gong, Dihong and Li, Zhifeng and Liu, Changsong and Liu, Wei},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  title={Occlusion Robust Face Recognition Based on Mask Learning With Pairwise Differential Siamese Network}, 
  year={2019},
  volume={},
  number={},
  pages={773-782},
  doi={10.1109/ICCV.2019.00086}}
