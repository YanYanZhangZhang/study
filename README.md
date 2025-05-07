# SPH-Font

## Requirements

(Welcome to develop SFS-Font together.)

We recommend you to use [Anaconda](https://www.anaconda.com/) to manage your libraries.

- [Python](https://www.python.org/) 3.9* 
- [PyTorch](https://pytorch.org/) 2.0.* 
- [TorchVision](https://pypi.org/project/torchvision/)
- [OpenCV](https://opencv.org/)

## Data Preparation
Please prepare the corresponding font images converted from [TTF fonts](https://www.foundertype.com/) in ./datasets-fine 

### Preparing the images
* The images are should be placed in this format:
```
    * dataroot
    |-- train
        |-- chinese
            |-- font1
                |-- char1.png
                |-- char2.png
                |-- char3.png
            |-- font2
                |-- char1.png
                |-- char2.png
                |-- char3.png
                    .
                    .
                    .
        |-- source
            |-- font1
                |-- char1.png
                |-- char2.png
                |-- char3.png
            |-- font2
                |-- char1.png
                |-- char2.png
                |-- char3.png
                        .
                        .
                        .

```
## Training

### Chinese font generation
```bash
 bash train.sh
```

## Testing

### Chinese font generation
```bash
 bash test.sh
```
