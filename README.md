# GP-SFH
--------

Gaussian process (GP) based implementation for galaxy star formation histories (SFHs) with physically motivated kernels.

Based on work in [Iyer & Speagle et al. 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220805938I/abstract); (accepted for publication in ApJ).

### Getting started:

The base class for implementing a GP-SFH instance can be initialized using

```python
>>> import numpy as np
>>> from gp_sfh_kernels import *

>>> zval = 0.1
>>> case = simple_GP_sfh(sp=sp, cosmo=cosmo, zval = zval)
```

Here, the sp object is a FSPS StellarPopulation instance that can be used to generate spectra from our SFHs, and cosmo is an astropy.cosmology instance.

Now, to use the `simple_GP_sfh` framework, we need to assign it a kernel, or covariance matrix. A few are available in the `gp_sfh_kernels.py` file, notably the Extended Regulator model kernel used in the paper. To use this kernel, we can do the following:

```python
>>> nsamp = 1000 # Number of SFH samples
>>> tarr_res = 1000 # resolution of the time array
>>> zval = 0.1 # redshift to compute observables at
>>> random_seed = 42
>>> case.get_tarr(n_tarr = tarr_res
>>> case.get_basesfh(sfhtype='const')

>>> kernel_params = [sig_reg, tau_in, tau_eq, sig_dyn, tau_dyn]
>>> case.kernel = extended_regulator_model_kernel_paramlist
```

The parameters here correspond to the two normalization factors for the regulator model and the dynamical component of the ACF, and three *effective* timescales corresponding to the inflow, equilibrium, and dynamical processes that affect the overall star formation in galaxies.

We are all set up. We can now draw samples from the GP and run them through FSPS to generate spectra:

```python
>>> case.samples = case1.sample_kernel(nsamp = nsamp, random_seed = random_seed, force_cov=True, kernel_params = kernel_params)
>>> case.get_spec(nsamp = nsamp)
>>> case.calc_spectral_features(massnorm = True)
```

All the figures in the paper can be recreated using code available in the `GP-SFH - all figures.ipynb` colab notebook.

### Usage:

This is not a full-fledged python package, so for the basic functionality just copy the `gp_sfh.py` and `gp_sfh_kernels.py` files to your working directory and you're good to go. You might also need to install FSPS & python-FSPS if you don't have that already, and dense basis if you want to implement variable base SFHs using that method.

If you have any problems installing or using GP-SFH, or would like to see any features not currently included, contact us or [raise an issue](https://github.com/kartheikiyer/GP-SFH/issues).

If you use this in your work, please cite [Iyer & Speagle et al. 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220805938I/abstract).

Get in touch!
- kgi2103@columbia.edu
- j.speagle@utoronto.ca

### Code acknowledgements:

- Base packages: numpy, astropy, tqdm, time, pickle, importlib
- Spectral modeling: FSPS, dense basis
- Plotting: matplotlib, seaborn, corner, chainconsumer
