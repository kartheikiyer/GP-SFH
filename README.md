# GP-SFH
--------

Gaussian process (GP) based implementation for galaxy star formation histories (SFHs) with physically motivated kernels.

Based on work in Speagle & Iyer et al. (2022; submitted to ApJ).

Getting started:

The base class for implementing a GP-SFH instance can be initialized using

```python
>>> import numpy as np
>>> from gp_sfh_kernels import *
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

We are all set up. We can now draw samples from the GP and run them through FSPS to generate spectra:

```python
>>> case.samples = case1.sample_kernel(nsamp = nsamp, random_seed = random_seed, force_cov=True, kernel_params = kernel_params)
>>> case.get_spec(nsamp = nsamp)
>>> case.calc_spectral_features(massnorm = True)
```

All the figures in the paper can be recreated using code available in the `GP-SFH - all figures.ipynb` colab notebook.

Get in touch!
- kartheik.iyer@dunlap.utoronto.ca
- j.speagle@utoronto.ca

Acknowledgements:
- numpy
- astropy
- FSPS
- dense basis
- Plotting: matplotlib, seaborn, corner, chainconsumer
