## Python sources for the method proposed in
# Synthetic Hyperspectral Images with Controllable Spectral Variability and Ground Truth

The code uses the Pytorch and Pytorch Lightning frameworks. The requirements needed to run the code is in the file requirements.txt. The method requires the hyperspectral images to be in Matlab mat files having the following named variables:

| Variable | Content |
| --- | ----------- |
| Y | Array having dimensions B x P containing the spectra |
| GT | Array having dimensions R x B containing the reference endmembers |
|cols | The number of columns in the hyperspectral image (HSI) |
|rows | The number of rows in the HSI |

Here, R is the number of endmembers, B the number of bands, and P the number of pixels. Edit the beta_vae.yaml file under the configs directory to change parameters for the beta_vae. Run the file train_vae.py to train the model. Edit and run the file run.py to generate synthetic HSIs. If you use the sources provided, make sure you cite the paper:

B. Palsson, M. O. Ulfarsson and J. R. Sveinsson, "Synthetic Hyperspectral Images With Controllable Spectral Variability and Ground Truth," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 6007205, doi: 10.1109/LGRS.2022.3150245.

@ARTICLE{9707767,  author={Palsson, Burkni and Ulfarsson, Magnus O. and Sveinsson, Johannes R.},  journal={IEEE Geoscience and Remote Sensing Letters},   title={Synthetic Hyperspectral Images With Controllable Spectral Variability and Ground Truth},   year={2022},  volume={19},  number={},  pages={1-5},  doi={10.1109/LGRS.2022.3150245}}
