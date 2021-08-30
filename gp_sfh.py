# main functions for the GP-SFH module.
# contents:
#   class simple_GP_sfh()

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.special as ssp

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import seaborn as sns
sns.set(font_scale=1.8)
sns.set_style('white')

try:
    import fsps
except:
    print('Failed to load FSPS. Install if spectral generation modules are needed.')

# Creating a base class that simplifies a lot of things.
# The way this is set up, you can pass a kernel as an argument
# to compute the covariance matrix and draw samples from it.

class simple_GP_sfh():

    """
    A class that creates and holds information about a specific
    kernel, and can generate samples from it.

    Attributes
    ----------
    tarr: fiducial time array used to draw samples
    kernel: accepts an input function as an argument,
        of the format:

            def kernel_function(delta_t, **kwargs):
                ... function interior ...
                return kernel_val[array of len(delta_t)]

    Methods
    -------
    get_covariance_matrix
        [although this has double for loops for maximum flexibility
        with generic kernel functions, it only has to be computed once,
        which makes drawing random samples super fast once it's computed.]
    sample_kernel
    plot_samples
    plot_kernel
    [to-do] condition on data

    """

    def __init__(self, sp = 'none', cosmo = cosmo, zval = 0.1):


        self.kernel = []
        self.covariance_matrix = []
        self.zval = zval
        self.sp = sp
        self.cosmo = cosmo
        self.get_t_univ()
        self.get_tarr()

    def init_SPS():

        mocksp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,sfh=0, imf_type=1, logzsol=0.0, dust_type=2, dust2=0.0, add_neb_emission=True)
        self.sp = mocksp
        return

    def make_MS_SFH(self, Mseed, timeax = np.arange(0,cosmo.age(0.0).value, 1e-3)):

        # following the right skew peak function parametrization from Ciesla+17
        # www.arxiv.org/pdf/1706.08531.pdf

        # Mseed [int] is the seed mass of the SFH at z=5
        # timeax [int, array] is any array of times along which the SFH is computed

        Ap = 6e-3
        taup = -0.84
        A = Ap*np.exp(-np.log10(Mseed)/taup)

        Ap = 47.39
        taup = 3.12
        mu = Ap*np.exp(-np.log10(Mseed)/taup)

        Ap = 17.08
        taup = 2.96
        sigma = Ap*np.exp(-np.log10(Mseed)/taup)

        slope = -0.56
        norm = 7.03
        rs = slope*np.log10(Mseed) + norm

        sfh = A * (np.pi/2) * (sigma) * np.exp( -((timeax-mu)/rs) + (sigma/(2*rs))**2) * ssp.erfc( (sigma/(2*rs)) - ((timeax-mu)/sigma))

        return sfh


    def get_basesfh(self,sfhtype='const', mstar = None):

        if sfhtype == 'const':
            self.basesfh = np.ones_like(self.tarr)* 1.0
        elif sfhtype == 'MS':
            sfh = self.make_MS_SFH(10**mstar, self.tarr)
            self.basesfh = np.log10(sfh)
        else:
            print('unknown basesfh type. set yourself with len of tarr.')
        return


    def get_t_univ(self):

        self.t_univ = self.cosmo.age(self.zval).value
        return

    def get_tarr(self, n_tarr = 1000):

        self.get_t_univ()
        if n_tarr > 1:
            self.tarr = np.linspace(0,self.t_univ, n_tarr)
        elif n_tarr < 1:
            self.tarr = np.arange(0,self.t_univ, n_tarr)
        else:
            raise('Undefined n_tarr: expected int or float.')
        return


    def get_covariance_matrix(self, **kwargs):
        """
        Evaluate covariance matrix with a particular kernel
        """

        cov_matrix = np.zeros((len(self.tarr),len(self.tarr)))
        for i in tqdm(range(len(cov_matrix))):
            for j in range(len(cov_matrix)):
                    cov_matrix[i,j] = self.kernel(self.tarr[i] - self.tarr[j], **kwargs)

        return cov_matrix

    def sample_kernel(self, nsamp = 100, random_seed = 42, force_cov=False, **kwargs):

        mean_array = np.zeros_like(self.tarr)
        if (len(self.covariance_matrix) == 0) or (force_cov == True):
            self.covariance_matrix = self.get_covariance_matrix(**kwargs)
#         else:
#             print('using precomputed covariance matrix')

        np.random.seed(random_seed)
        samples = np.random.multivariate_normal(mean_array,self.covariance_matrix,size=nsamp)

        return samples

    def get_spec(self, nsamp):

        bands = fsps.list_filters()
        filter_wavelengths = [fsps.filters.get_filter(bands[i]).lambda_eff for i in range(len(bands))]

        all_lam, all_spec, all_spec_massnorm, all_mstar, all_emline_wav, all_emline_lum, all_emline_lum_massnorm, all_filtmags = [], [], [], [], [], [], [], []

        for i in tqdm(range(nsamp)):
            specsfh = 10**(self.basesfh+self.samples[i, 0:])
            self.sp.set_tabular_sfh(self.tarr, specsfh)
            lam, spec = self.sp.get_spectrum(tage = self.t_univ)
            mstar = self.sp.stellar_mass
            bandmags = self.sp.get_mags(tage = self.cosmo.age(self.zval).value, redshift = self.zval, bands = bands)

            all_lam.append(lam)
            all_spec.append(spec)
            all_spec_massnorm.append(spec/mstar)
            all_mstar.append(mstar)
            all_emline_wav.append(self.sp.emline_wavelengths)
            all_emline_lum.append(self.sp.emline_luminosity)
            all_emline_lum_massnorm.append(self.sp.emline_luminosity / mstar)
            all_filtmags.append(bandmags)

        self.lam = all_lam
        self.spec = all_spec
        self.spec_massnorm = all_spec_massnorm
        self.mstar = all_mstar
        self.emline_wav = all_emline_wav
        self.emline_lum = all_emline_lum
        self.emline_lum_massnorm = all_emline_lum_massnorm

        # not mass normalized
        self.bands = bands
        self.filter_wavelengths = filter_wavelengths
        self.filtmags = all_filtmags

        return

    def calc_spectral_features(self, massnorm = True):

        if massnorm == True:
            spectra = self.spec_massnorm
            emline_lum = self.emline_lum_massnorm
        else:
            spectra = self.spec
            emline_lum = self.emline_lum

        lam = self.lam[0]
        emline_wavs = self.emline_wav

        ha_lums = []
        hdelta_ews = []
        dn4000_vals = []

        ha_lambda = 6562 # in angstrom

        for i in tqdm(range(len(spectra))):

            ha_line_index = np.argmin(np.abs(emline_wavs[i] - ha_lambda))
            ha_lum = emline_lum[i][ha_line_index]
            ha_lums.append(ha_lum)

            hdelta_mask = (lam > 4041.60) & (lam < 4079.75)
            #hdelta_cont1_flux = np.trapz(x=lam[hdelta_mask], y = spectra[i][hdelta_mask])
            hdelta_cont1_flux = np.mean(spectra[i][hdelta_mask])
            hdelta_mask = (lam > 4128.50) & (lam < 4161.00)
            #hdelta_cont2_flux = np.trapz(x=lam[hdelta_mask], y = spectra[i][hdelta_mask])
            hdelta_cont2_flux = np.mean(spectra[i][hdelta_mask])
            hdelta_cont_flux_av = (hdelta_cont1_flux + hdelta_cont2_flux)/2

            hdelta_mask = (lam > 4083) & (lam < 4122)
            #hdelta_emline_flux = np.trapz(x=lam[hdelta_mask], y = spectra[i][hdelta_mask])
            hdelta_emline_fluxes = spectra[i][hdelta_mask]
            hdelta_emline_fluxratios = hdelta_emline_fluxes / hdelta_cont_flux_av

            hdelta_ew = np.sum(1 - hdelta_emline_fluxratios)
            hdelta_ews.append(hdelta_ew)

            dn4000_mask = (lam>3850) & (lam < 3950)
            dn4000_flux1 = np.mean(spectra[i][dn4000_mask])
            dn4000_mask = (lam>4000) & (lam < 4100)
            dn4000_flux2 = np.mean(spectra[i][dn4000_mask])
            dn4000 = dn4000_flux1/dn4000_flux2
            dn4000_vals.append(dn4000)

            self.ha_lums = ha_lums
            self.hdelta_ews = hdelta_ews
            self.dn4000_vals = dn4000_vals

        return


    def plot_samples(self, nsamp = 100, random_seed = 42, plot_samples=5,plim=2, plotlog=False,save_fname = 'none', **kwargs):

        samples = self.sample_kernel(nsamp = nsamp, random_seed = random_seed, **kwargs)

        plt.figure(figsize=(12,6))
        if plotlog == True:
            plt.plot(self.tarr, 10**samples.T[0:,0:plot_samples],'-',alpha=0.7,lw=1)
            plt.plot(self.tarr, 10**np.nanpercentile(samples.T,50,axis=1),'k',lw=3,label='median')
            plt.xlabel('time [arbitrary units]')
            plt.ylabel('some quantity of interest')
        else:
            plt.plot(self.tarr, samples.T[0:,0:plot_samples],'-',alpha=0.7,lw=1)
            plt.plot(self.tarr, np.nanpercentile(samples.T,50,axis=1),'k',lw=3,label='median')
            plt.fill_between(self.tarr, np.nanpercentile(samples.T,16,axis=1),
                            np.nanpercentile(samples.T,84,axis=1),color='k',alpha=0.1,label='1$\sigma$')
            plt.xlabel('time [arbitrary units]')
            plt.ylabel('some quantity of interest')
            plt.legend(edgecolor='w');
            plt.ylim(-plim,plim);plt.title([kwargs])

        if save_fname is not 'none':
            print('saving figure as: ',save_fname)
            plt.savefig(save_fname, bbox_inches='tight')
        plt.show()

    def plot_kernel(self, deltat = np.round(np.arange(-10,10,0.1),1),save_fname = 'none', **kwargs):

        plt.figure(figsize=(12,6))
        plt.plot(deltat, self.kernel(deltat, **kwargs),lw=3,
                 label=kwargs)
        plt.xlabel('$\Delta t$')
        plt.ylabel('covariance');plt.title(kwargs)
        #plt.text(-9,0.23,'Past');plt.text(7,0.23,'Future')
        if save_fname is not 'none':
            print('saving figure as: ',save_fname)
            plt.savefig(save_fname, bbox_inches='tight')
        plt.show()

    def plot_kernel_and_draws(self, deltat = np.round(np.arange(-10,10,0.1),1), nsamp = 100, random_seed = 42, plot_samples=5,plim=2, plotlog=False,save_fname = 'none', **kwargs):


        plt.figure(figsize=(24,6))

        plt.subplot(1,2,1)
        plt.plot(deltat, self.kernel(deltat, **kwargs),lw=3,
                 label=kwargs)
        plt.xlabel('$\Delta t$')
        plt.ylabel('covariance');plt.title(['(kernel)',kwargs])
        plt.xlim(-np.amax(self.tarr),np.amax(self.tarr));
        #plt.text(-9,0.23,'Past');plt.text(7,0.23,'Future')

        plt.subplot(1,2,2)

        samples = self.sample_kernel(nsamp = nsamp, random_seed = random_seed, **kwargs)

        if plotlog == True:
            plt.plot(self.tarr, 10**samples.T[0:,0:plot_samples],'-',alpha=0.7,lw=1)
            plt.plot(self.tarr, 10**np.nanpercentile(samples.T,50,axis=1),'k',lw=3,label='median')
            plt.xlabel('time [arbitrary units]')
            plt.ylabel('log SFR(t)')
        else:
            plt.plot(self.tarr, samples.T[0:,0:plot_samples],'-',alpha=0.7,lw=1)
            plt.plot(self.tarr, np.nanpercentile(samples.T,50,axis=1),'k',lw=3,label='median')
            plt.fill_between(self.tarr, np.nanpercentile(samples.T,16,axis=1),
                            np.nanpercentile(samples.T,84,axis=1),color='k',alpha=0.1,label='1$\sigma$')
            plt.xlabel('time [arbitrary units]')
            plt.ylabel('log SFR(t)')
            plt.legend(edgecolor='w');
            plt.ylim(-plim,plim);
        plt.title('samples drawn from kernel')

        if save_fname is not 'none':
            print('saving figure as: ',save_fname)
            plt.savefig(save_fname, bbox_inches='tight')
        plt.show()

    def plot_kernel_sfhs_spec(self, deltat = np.round(np.arange(-10,10,0.1),1), nsamp = 100, random_seed = 42, plot_samples=5,plim=2, plotlog=False,save_fname = 'none', titlestr = '', massnorm = True, **kwargs):

        plt.figure(figsize=(24,6))

        plt.subplot(1,3,1)
        plt.plot(deltat, self.kernel(deltat, **kwargs),lw=3,
                 label=kwargs)
        plt.xlabel('$\Delta t$ [Gyr]')
        plt.ylabel('covariance');#plt.title(['(kernel)',kwargs])
        plt.title(titlestr)
        plt.xlim(-np.amax(self.tarr),np.amax(self.tarr));
        #plt.text(-9,0.23,'Past');plt.text(7,0.23,'Future')

        plt.subplot(1,3,2)

        samples = self.sample_kernel(nsamp = nsamp, random_seed = random_seed, **kwargs)

        if plotlog == True:
            plt.plot(self.tarr, 10**samples.T[0:,0:plot_samples],'-',alpha=0.7,lw=1)
            plt.plot(self.tarr, 10**np.nanpercentile(samples.T,50,axis=1),'k',lw=3,label='median')
            plt.xlabel('time [arbitrary units]')
            plt.ylabel('log SFR(t)')
        else:
            plt.plot(self.tarr, samples.T[0:,0:plot_samples],'-',alpha=0.7,lw=1)
            plt.plot(self.tarr, np.nanpercentile(samples.T,50,axis=1),'k',lw=3,label='median')
            plt.fill_between(self.tarr, np.nanpercentile(samples.T,16,axis=1),
                            np.nanpercentile(samples.T,84,axis=1),color='k',alpha=0.1,label='1$\sigma$')
            plt.xlabel('time [Gyr]')
            plt.ylabel('log SFR(t)')
            plt.legend(edgecolor='w');
            plt.ylim(-plim,plim);
            #plt.xlim(0,1)
        plt.title('samples drawn from kernel')

        plt.subplot(1,3,3)

        for i in range(plot_samples):
            specsfh = 10**(self.basesfh+samples[i, 0:])
            self.sp.set_tabular_sfh(self.tarr, specsfh)
            lam, spec = self.sp.get_spectrum(tage = self.t_univ)
            mstar = self.sp.stellar_mass

            if massnorm == True:
                plt.plot(lam, spec/mstar*1e10,alpha=0.7,lw=1)
            else:
                plt.plot(lam, spec,alpha=0.7,lw=1)

        plt.xscale('log');plt.yscale('log')
        plt.xlabel(r'$\lambda$ [rest-frame]')
        plt.ylabel(r'$L_\nu$ [L$_\odot~/~$Hz]')
        #plt.ylim(1e-6,1e-1)
        plt.xlim(1e3,1e5)

        plt.tight_layout()
        if save_fname is not 'none':
            print('saving figure as: ',save_fname)
            plt.savefig(save_fname, bbox_inches='tight')
        plt.show()