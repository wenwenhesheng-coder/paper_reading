# ROMAN Robust PCA Masking Network

Pytorch implementation of the ROMAN-S and ROMAN-R models.

Code of our paper: "Interpretable Neural Networks for Video Separation: Deep Unfolding RPCA with Foreground Masking".

- [ ] B. Joukovsky, Y. C. Eldar and N. Deligiannis, "Interpretable Neural Networks for Video Separation: Deep Unfolding RPCA With Foreground Masking," in IEEE Transactions on Image Processing, vol. 33, pp. 108-122, 2024, doi: 10.1109/TIP.2023.333617


## Dataset

The code was trained and tested on various videos form the [CDNet2014](http://changedetection.net/) dataset. To train your own models, please download the dataset and set the correct data_path to your local dataset root


## Training

### ROMAN-R and ROMAN-S

The following command line can be used to train ROMAN-R on the baseline/highway video sequence:

```
python main_roman.py --category 2 --log-dir ./logs/baseline/highway/ --coeff-L .8 --coeff-S .05 --coeff-Sside .025 --split 0  --model roman_r --hidden-filters 32 --initial-lr 0.003 --loss_type L_tversky_bce --reweighted --layers 3
```

Check the `configs/train.sh` files for other examples on other categories and `configs/cdnet.py` for other options.

### The 3D UNET baseline

```
python main_roman.py --category 2 --log-dir ./logs/baseline/highway/ --split 0 --model unet3d --initial-lr 0.001 --loss_type tversky_cce
```

Pretrained weights can be found here [unet_carvana_scale1.0_epoch2.pth](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0).


## Testing

Once a model has been trained, it can be tested using `eval.py` to compute detection scores according to the CDNet2014 dataset guidelines, on entire video clips. Use the following:

```
python eval.py -c baseline/highway -p <path to directory containing the model checkpoint and yaml configuration>
```

