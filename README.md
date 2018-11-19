# Estimate Outdoor Illumination on Dash-cam images #
A re-implemented project which focus on dash-cam images
according to the paper (Deep Outdoor Illumination Estimation [Hold-Geoffroy et al. CVPR 2017]). 
This project is an end-to-end system that outputs corresponding sun position and physcial 
sky, camera parameters by inputing single dash-cam image.

<img src="teaser.png" width="700" />

## Quick start ##
### Test ###
If you want to test your own image, run this command:
```bash
python inference.py --img_path <image-path>
```

### Training ###
You can generate the dataset and list by using ```generate_data.py ``` and 
the data (360 panorama images seperated into test and train) which followed the format in ```GS_skymodel.csv```.

After generating dataset, run command below for training:
```bash
python train.py
```
The trained weights will be stored in ''pre-trained'' folder (**It will replace original weights!**)

### Evaluation ###
Evaluate the trained model by executing ```eval.py```, it will output the average error of each predictions.

## Dependecies ##
* numpy
* skimage
* pytorch
* opencv-python
* progressbar
