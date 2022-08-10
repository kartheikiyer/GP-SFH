# The gp_sfh.py file conta"ins a simple GP framework that we'll be using
from gp_sfh import *
import gp_sfh_kernels

from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from copy import deepcopy
import pickle
import seaborn as sns
sns.set(font_scale=1.4)
sns.set_style('white')
import pandas as pd
import corner

sig_reg = np.sqrt(0.97) * (0.4)
sig_dyn = np.sqrt(0.03) * (0.4)
kernel_params_MW_1dex = [sig_reg, 2500/1e3, 150/1e3, sig_dyn, 25/1e3]
kernel_params_dwarf_1dex = [sig_reg, 30/1e3, 150/1e3, sig_dyn, 10/1e3]
kernel_params_noon_1dex = [sig_reg, 200/1e3, 100/1e3, sig_dyn, 50/1e3]
kernel_params_highz_1dex = [sig_reg, 15/1e3, 16/1e3, sig_dyn, 6/1e3]
print('using $\sigma_{reg}: %.4f$, $\sigma_{dyn}: %.4f$. ' %(sig_reg, sig_dyn))

TCF20_scattervals = [0.17, 0.53, 0.24, 0.27]
regnorm = 0.4
kernel_params_MW_TCF20 = [sig_reg/regnorm * TCF20_scattervals[0], 2500/1e3, 150/1e3, sig_dyn/regnorm * TCF20_scattervals[0], 25/1e3]
kernel_params_dwarf_TCF20 = [sig_reg/regnorm * TCF20_scattervals[1], 30/1e3, 150/1e3, sig_dyn/regnorm * TCF20_scattervals[1], 10/1e3]
kernel_params_noon_TCF20 = [sig_reg/regnorm * TCF20_scattervals[2], 200/1e3, 100/1e3, sig_dyn/regnorm * TCF20_scattervals[2], 50/1e3]
kernel_params_highz_TCF20 = [sig_reg/regnorm * TCF20_scattervals[3], 15/1e3, 16/1e3, sig_dyn/regnorm * TCF20_scattervals[3], 6/1e3]

cases = ['MW','dwarf','noon','highz']
case_names = ['MWA','Dwarf','Noon','High-z']
case_params = [kernel_params_MW_1dex, kernel_params_dwarf_1dex, kernel_params_noon_1dex, kernel_params_highz_1dex]
case_colors = ['crimson','darkorange','seagreen','deepskyblue']

def makeplot(fig, axs, axins, case1, kernelcolor, labelval, **kernelargs):

    axs[0].plot(case1.tarr, case1.kernel(case1.tarr, **kernelargs),lw=5,color=kernelcolor,label=labelval,zorder=100)
    axs[0].set_xlabel('$|\Delta (t-t\')|$ [Gyr]')
    axs[0].set_ylabel('ACF [(dex)$^2$/Myr]')
    axs[0].set_xscale('log')
    axs[0].set_xlim(np.amin(case1.tarr),np.amax(case1.tarr))
    
    tmep = case1.samples[0]
    
    #tmep = tmep/np.std(tmep)*0.6
#     axs[1].plot(case1.tarr, case1.samples[0],color=kernelcolor)
    axs[1].plot(case1.tarr, tmep,color=kernelcolor)
    axs[1].set_xlim(0,np.amax(case1.tarr));
    axs[1].set_ylim(-4,3)
    axs[1].set_xlabel('time [Gyr]')
    axs[1].set_ylabel('log SFR(t) [M$_\odot$yr$^{-1}$]')
    
    #axins.plot(case1.tarr, case1.samples[0],color=kernelcolor)
    axins.plot(case1.tarr, tmep,color=kernelcolor)
    # sub region of the original image
    x1, x2, y1, y2 = 2.5,3.0, -1.08,1.08
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    #axins.set_yticklabels('')
    axs[1].indicate_inset_zoom(axins, edgecolor="black")

def make_kernel_plot(casevals, cases, save_str):

    fig, axs = plt.subplots(1,2,figsize=(10*1.5,5*1.5))
    axins = axs[1].inset_axes([0.18, 0.05, 0.8, 0.3])
    
    case1 = casevals[0]

    axs[1].plot(case1.tarr, np.zeros_like(case1.tarr),'k',alpha=0.3)
    axins.plot(case1.tarr, np.zeros_like(case1.tarr),'k',alpha=0.3)

    axs[1].fill_between(case1.tarr, np.ones_like(case1.tarr),np.ones_like(case1.tarr)*(-1.),color='k',alpha=0.1)

    for i, case in enumerate(cases):

        case1 = casevals[i]
        makeplot(fig, axs, axins, case1, kernelcolor=case_colors[i], labelval=case, 
                                         kernel_params = case_params[i])

    
    axs[0].legend(edgecolor='w',loc=1,framealpha=0.1)
    axs[0].set_yscale('log');axs[0].set_ylim(1e-2,1e0)
    plt.tight_layout()
    plt.savefig('figures/fig3_panel1_'+save_str+'.png',bbox_inches='tight')
    plt.show()
    
def make_sfh_draws_plot(casevals, cases, save_str):

    fig, axs = plt.subplots(2,2,figsize=(14,10))
    plt.subplots_adjust(wspace=0, hspace=0.2)
    axins = []
    var_sigmas = []
    
    case1 = casevals[0]
    for i, case in enumerate(cases):

        axins.append(axs[int(i/2),int(i%2)].inset_axes([0.18, 0.05, 0.8, 0.3]))
        axs[int(i/2),int(i%2)].plot(case1.tarr, np.zeros_like(case1.tarr),'k',alpha=0.3)
        axins[i].plot(case1.tarr, np.zeros_like(case1.tarr),'k',alpha=0.3)
        if save_str == 'var':
            axs[int(i/2),int(i%2)].fill_between(case1.tarr, np.ones_like(case1.tarr)*TCF20_scattervals[i],np.ones_like(case1.tarr)*(-1. * TCF20_scattervals[i]),color='k',alpha=0.1)
        else:
            axs[int(i/2),int(i%2)].fill_between(case1.tarr, np.ones_like(case1.tarr)*0.4,np.ones_like(case1.tarr)*(-0.4),color='k',alpha=0.1)

        case1 = casevals[i]
        np.random.seed(12)
        for j in range(5):
            tempalpha = np.random.uniform()
            tmep = case1.samples[j]
            axs[int(i/2),int(i%2)].plot(case1.tarr, tmep,color=case_colors[i],alpha=tempalpha)
            axins[i].plot(case1.tarr, tmep,color=case_colors[i],alpha=tempalpha)

        axs[int(i/2),int(i%2)].set_xlim(0,np.amax(case1.tarr));
        axs[int(i/2),int(i%2)].set_ylim(-2.7,1.4)
        if (i == 0) or (i==2):
            axs[int(i/2),int(i%2)].set_ylabel('log SFR(t) [M$_\odot$yr$^{-1}$]')
        else:
            axs[int(i/2),int(i%2)].set_yticks([])
        if (i>1):
            axs[int(i/2),int(i%2)].set_xlabel('time [Gyr]')
            axs[int(i/2),int(i%2)].set_xticks(np.arange(0,12,2))

        # sub region of the original image
        x1, x2, y1, y2 = 2.5,3.0, -1.08,1.08
        axins[i].set_xlim(x1, x2)
        axins[i].set_ylim(y1, y2)
        axins[i].set_xticklabels('')
        #axins.set_yticklabels('')
        axs[int(i/2),int(i%2)].indicate_inset_zoom(axins[i], edgecolor="black")

    plt.savefig('figures/fig3_panel2_'+save_str+'.png',bbox_inches='tight')
    plt.show()

# plot_line(912, 'Lyman limit')
# plot_line(972.54, r'Ly-$\gamma$')
# plot_line(1025.72, r'Ly-$\beta$')
# plot_line(1031.93,'OVI')
# plot_line(1037.62,'OVI(2xweaker)')
# plot_line(1144.94,'FeII')
# plot_line(1190.41,'SiII(blends1192)')
# plot_line(1193.29,'SiII(2xstronger)')
# plot_line(1215.67, r'Ly-$\alpha$')
# plot_line(1238.82,'NV(blends1240)')
# plot_line(1242.80,'NV')
# plot_line(1260.42,'SiII')
# plot_line(1302.17,'OI')
# plot_line(1304.37,'SiII(10xweakerthan1260)')
# plot_line(1334.53,'CII')
# plot_line(1640.47,'HeII')
# plot_line(1486,'NIV]')
# plot_line(1526.71,'SiII')
# plot_line(1550.77,'CIV') # em-and-abs
# plot_line(2796.35,'MgII')
# plot_line(2852,'MgI')
# plot_line(3203.1,'HeII')
# plot_line(3312.3,'OIII') # absorption
# plot_line(3444.1,'OIII')
# plot_line(3726.2,'[OII]') # em
# plot_line(3728.9,'[OII]')
# plot_line(3797.9,'H8')
# plot_line(3835.3,'H7')
# plot_line(3869,'[NeIII]')
# plot_line(3889.1,'H6')
# plot_line(3933.66,'CaII_K')
# plot_line(3968.47,'CaII_H+H5, [NeIII]')
# plot_line(4101.74,'Hdelta')
# plot_line(4305.,'G_band')
# plot_line(4340.46,'Hgamma')
# plot_line(4363,'[OIII]')
# plot_line(4396,'He I')
# plot_line(4471,'He I')
# plot_line(4686,'He II')
# plot_line(4861.33,'Hbeta')
# plot_line(4958.91,'[OIII]')
# plot_line(5006.8,'[OIII]') #stronger
# plot_line(5167.32,'MgI')
# plot_line(5172.68,'MgI(blends5174.5)')
# plot_line(5183.60,'MgI')
# plot_line(5648.,'[NII]')
# plot_line(5876,'HeI')
# plot_line(5889.95,'NaI')
# plot_line(6300,'[OI]')
# plot_line(6548,'[NII]')
# plot_line(6562.82,'Halpha')
# plot_line(6584.,'[NII]')
# plot_line(6678,'HeI')
# plot_line(6717.0,'[SII](blends6724)')
# plot_line(6731.3,'[SII]')
    
def ujy_to_flam(data,lam):
    flam = ((3e-5)*data)/((lam**2.)*(1e6))
    return flam/1e-19

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
    fuv_vals = []
    nuv_vals = []
    u_vals = []
    caH_ews = []
    caK_ews = []

    ha_lambda = 6562 # in angstrom

    for i in (range(len(spectra))):
        
#         specflam = ujy_to_flam(spectra[i], lam)
        specflam = spectra[i]

        ha_line_index = np.argmin(np.abs(emline_wavs[i] - ha_lambda))
        ha_lum = emline_lum[i][ha_line_index]
        ha_lums.append(ha_lum)

#         hdelta_mask = (lam > 4041.60) & (lam < 4079.75)
        hdelta_mask = (lam > 4030.) & (lam < 4082.)
        #hdelta_cont1_flux = np.trapz(x=lam[hdelta_mask], y = spectra[i][hdelta_mask])
        hdelta_cont1_flux = np.mean(specflam[hdelta_mask])
#         hdelta_mask = (lam > 4128.50) & (lam < 4161.00)
        hdelta_mask = (lam > 4122.0) & (lam < 4170.00)
        #hdelta_cont2_flux = np.trapz(x=lam[hdelta_mask], y = spectra[i][hdelta_mask])
        hdelta_cont2_flux = np.mean(specflam[hdelta_mask])
        hdelta_cont_flux_av = (hdelta_cont1_flux + hdelta_cont2_flux)/2

        hdelta_mask = (lam > 4083.5) & (lam < 4122.5)
#         hdelta_mask = (lam > 4095.5) & (lam < 4109.5)
        hdelta_emline_fluxes = np.trapz(x=lam[hdelta_mask], 
                                        y = (hdelta_cont_flux_av - specflam[hdelta_mask])/hdelta_cont_flux_av)
        
#         hdelta_emline_fluxes = (specflam[hdelta_mask])
        hdelta_emline_fluxratios = hdelta_emline_fluxes / hdelta_cont_flux_av

#         hdelta_ew = np.sum(1 - hdelta_emline_fluxratios)
        hdelta_ew = hdelta_emline_fluxes
        hdelta_ews.append(hdelta_ew)
        
        lam_index_caK = np.argmin(np.abs(self.lam[0] - 3933.66))
        lam_index_caH = np.argmin(np.abs(self.lam[0] - 3968.47))
        
        caK_mask = (lam > 3907.0064) & (lam <  3929.5122)
        caK_cont1_flux = np.mean(specflam[caK_mask])
        caK_mask = (lam > 3941.2155) & (lam < 3961.0205)
        caK_cont2_flux = np.mean(specflam[caK_mask])
        caK_cont_flux_av = (caK_cont1_flux + caK_cont2_flux)/2

        caK_mask = (lam > 3929.5122) & (lam < 3941.2155)
        caK_emline_fluxes = np.trapz(x=lam[caK_mask], 
                                y = (caK_cont_flux_av - specflam[caK_mask])/caK_cont_flux_av)
        caK_ew = caK_emline_fluxes
        caK_ews.append(caK_ew)
        
        caH_mask = (lam > 3941.2155) & (lam < 3961.0205)
        caH_cont1_flux = np.mean(specflam[caH_mask])
        caH_mask = (lam > 3980.8257) & (lam < 3997.0299)
        caH_cont2_flux = np.mean(specflam[caH_mask])
        caH_cont_flux_av = (caH_cont1_flux + caH_cont2_flux)/2

        caH_mask = (lam > 3961.0205) & (lam < 3980.8257)
        caH_emline_fluxes = np.trapz(x=lam[caH_mask], 
                                y = (caH_cont_flux_av - specflam[caH_mask])/caH_cont_flux_av)
        caH_ew = caH_emline_fluxes
        caH_ews.append(caH_ew)

        
#         if i<10:
#             plt.plot(lam[(lam>4030) & (lam<4170)], specflam[(lam>4030) & (lam<4170)])
#             plt.plot(lam[hdelta_mask], specflam[hdelta_mask])
#             plt.plot(lam[hdelta_mask], np.ones((np.sum(hdelta_mask)))*hdelta_cont_flux_av)
#             plt.show()
#             print(hdelta_ew)

        
#         specflam = spectra[i]
        dn4000_mask1 = (lam>3850) & (lam < 3950)
        dn4000_flux1 = np.mean(specflam[dn4000_mask1])
        dn4000_mask2 = (lam>4000) & (lam < 4100)
        dn4000_flux2 = np.mean(specflam[dn4000_mask2])
#         dn4000_mask1 = (lam>3850) & (lam < 3950)
#         dn4000_flux1 = np.mean(spectra[i][dn4000_mask1])
#         dn4000_mask2 = (lam>4000) & (lam < 4100)
#         dn4000_flux2 = np.mean(spectra[i][dn4000_mask2])
        dn4000 = dn4000_flux2/dn4000_flux1
        dn4000_vals.append(dn4000)
        
        fuv_lum_mask = (lam > 1300) & (lam < 1700)
        fuv_flux1 = np.mean(specflam[fuv_lum_mask])
        fuv_vals.append(fuv_flux1)
        
        nuv_lum_mask = (lam > 1800) & (lam < 2600)
        nuv_flux1 = np.mean(specflam[nuv_lum_mask])
        nuv_vals.append(nuv_flux1)
        
        u_lum_mask = (lam > 3000) & (lam < 3800)
        u_flux1 = np.mean(specflam[u_lum_mask])
        u_vals.append(u_flux1)
        
    return ha_lums, hdelta_ews, dn4000_vals, fuv_vals, nuv_vals, u_vals, caH_ews, caK_ews


def make_spectral_corner_plot(casevals, cases, save_str='fixed', smoothval=1.0):

    for i, case in reversed(list(enumerate(cases))):

        case1 = casevals[i]
        case1.calc_spectral_features(massnorm = True)
        ha_lums, hdelta_ews, dn4000_vals, fuv_vals, nuv_vals, u_vals, caH_ews, caK_ews = calc_spectral_features(case1)

        temphd = np.array(case1.hdelta_ews.copy())
        tempha = np.array(np.log10(ha_lums).copy())

        labels = [r'log H$\alpha$',
                  r'log F$_{\nu, NUV}$', 
                  r'H$\delta_{\rm EW}$', 
                  r'Ca-K$_{\rm EW}$', 
                  r'Ca-H$_{\rm EW}$',
                  r'log F$_{\nu, u}$', 
                  r'D$_n$(4000)', ]
        ndim = len(labels)
        case1_specfeatures = np.vstack((tempha, np.log10(nuv_vals), temphd,caK_ews, caH_ews, 
                                        np.log10(u_vals), case1.dn4000_vals))

        if i==3:
            fig = corner.corner(case1_specfeatures.T, 
                          labels=labels,
                          levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)], # quantiles=(0.16,0.84), 
                          plot_datapoints=False, fill_contours=True, smooth=smoothval,
                          color = case_colors[i], hist_kwargs={'lw':0, 'density':True})
            axes = np.array(fig.axes).reshape((ndim,ndim))
            all_specfeatures = case1_specfeatures

            for kdei in range(ndim):
                axkde = axes[kdei, kdei]
                sns.kdeplot(case1_specfeatures[kdei,0:],shade=True,lw=2,color=case_colors[i],ax=axkde)
                axkde.set_ylabel('')

        elif i<4:
            fig = corner.corner(case1_specfeatures.T, 
                  levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)], #quantiles=(0.16,0.84), 
                  plot_datapoints=False, fill_contours=True, smooth=smoothval,
                  color = case_colors[i], fig = fig, hist_kwargs={'lw':0, 'density':True})

            for kdei in range(ndim):
                axkde = axes[kdei, kdei]
                sns.kdeplot(case1_specfeatures[kdei,0:],shade=True,lw=2,color=case_colors[i],ax=axkde)
                axkde.set_ylabel('')
                temp_ylim = axkde.get_ylim()
                axkde.set_ylim(0,temp_ylim[1]*1.2)
                if i>=3:
                    axkde.set_ylim(0,temp_ylim[1]*1.4)
                elif i == 2:
                    axkde.set_ylim(0,temp_ylim[1]*1.08)
            all_specfeatures = np.hstack((all_specfeatures, case1_specfeatures))

    # Extract the axes
    lims = [(-3.2,-1.2),(-16.7,-15),(-8,8),(0.5,4.5),(-5,9),(-16.4,-14.4),(0.9,1.5)]

    for i in range(ndim):
        ax = axes[i, i]
        ax.set_xlim(lims[i])

        for j in range(i):
            ax = axes[i,j]
            ax.set_xlim(lims[j])
            ax.set_ylim(lims[i])
            
    plt.savefig('figures/spectral_corner_v2_'+save_str+'.png',bbox_inches='tight')
    plt.show()
    
    
def make_spectral_features_fig(temps, casevals, cases, spec_normed, save_str, plt_offset = 0.06, llim = -0.04, ulim = 4.2, line_y = 0.83):
    
    # Start the figure
    fig = plt.figure(figsize=(27*0.8,15*0.8))
    gs = GridSpec(1,5, figure=fig, width_ratios=[0.8,0.8,1,1,1])
    plt.subplots_adjust(wspace=0.03)

    # Full spectra - median
    ax1 = fig.add_subplot(gs[0,0:2])
    plt.fill_between([900,3000],[llim,llim], [ulim*plt_offset,ulim*plt_offset],color='k',alpha=0.1)
    plt.fill_between([3600,5250],[llim,llim], [ulim*plt_offset,ulim*plt_offset],color='k',alpha=0.1)
    plt.fill_between([5810,6800],[llim,llim], [ulim*plt_offset,ulim*plt_offset],color='k',alpha=0.1)

    # Going dwarf, noon, MW, high-z for better readability
    for i, ii in enumerate(np.array([0,2,3,1])):
        case1 = casevals[i]
        plt.axhline(0 + plt_offset*i,lw=3,color='k',alpha=0.3)
#         plt.plot(case1.lam[0], (temps[ii] - np.nanmedian(np.array(temps),0)) + plt_offset*i, color=case_colors[ii])
        plt.plot(case1.lam[0], -(temps[ii] - np.log10(spec_normed)) + plt_offset*i, color=case_colors[ii])
        if case_names[ii] == 'High-z':
            plt.text(2.5e5, 0+plt_offset*i + 0.015, case_names[ii],color=case_colors[ii],
                     fontsize=24, fontweight='bold',ha='right')
        else:
            plt.text(2.5e5, 0+plt_offset*i + 0.015, case_names[ii],color=case_colors[ii],
                     fontsize=24, fontweight='bold',ha='right')
#                 plt.fill_between(case1.lam[0], 
#                          (temps_hi[ii] - np.nanmedian(np.array(temps),0)) + plt_offset*i, 
#                          (temps_lo[ii] - np.nanmedian(np.array(temps),0)) + plt_offset*i, 
#                          color=case_colors[ii], alpha=0.1)

    plt.xscale('log');
    plt.xlim(5e2,3e5); plt.ylim(llim, ulim*plt_offset)
    plt.xlabel('rest-frame wavelength [$\AA$]')
    plt.ylabel('$\Delta$ Luminosity [Model-Ref (dex)] + offset')

    temp = [0,plt_offset/2,0,plt_offset/2,0,plt_offset/2]
    for i in np.arange(0,plt_offset*(ulim),plt_offset/2).ravel():
        temp.append(np.round(i,3))
    plt.yticks(np.arange(0,plt_offset*ulim, plt_offset/2),temp[0:len(np.arange(0,plt_offset*ulim, plt_offset/2))])

    ax2 = fig.add_subplot(gs[0,2])

    plot_line(912, 'Lyman limit', color='firebrick', line_y=line_y)
    plot_line(1031.93,'OVI', color='dodgerblue', line_y=line_y)
    plot_line(1215.67, r'Ly-$\alpha$', color='firebrick', line_y=line_y)
    plot_line(1550.77,'CIV', line_y=line_y) # em-and-abs
    plot_line(2796.35,'MgII', line_y=line_y)
    # plot_line(3312.3,'OIII', color='dodgerblue') # absorption

    for i, ii in enumerate(np.array([0,2,3,1])):
        plt.axhline(0 + plt_offset*i,lw=3,color='k',alpha=0.3)
#         plt.plot(case1.lam[0], (temps[ii] - np.nanmedian(np.array(temps),0)) + plt_offset*i, color=case_colors[ii])
        plt.plot(case1.lam[0], -(temps[ii] - np.log10(spec_normed)) + plt_offset*i, color=case_colors[ii])

    # plt.xscale('log');
    # plt.yticks(np.arange(-plt_offset/2,plt_offset*4.3, plt_offset/2), 
    #            [-plt_offset/2,0,plt_offset/2,0,plt_offset/2,0,plt_offset/2,0,plt_offset/2, plt_offset])
    plt.yticks([])
    plt.xlim(5e2,3e5); plt.ylim(llim, ulim*plt_offset)
    plt.xlabel('rest-frame wavelength [$\AA$]')
    # plt.ylabel('$\Delta$ Luminosity [dex] + offset')
    plt.xlim(0.81e3,3000)
    plt.minorticks_off()
    plt.xticks(np.arange(900,3000,600), np.round(np.arange(900,3000,600),0))
    plt.title(r'rest-UV')

    ax3 = fig.add_subplot(gs[0,3])

    plot_line(3726.2,'[OII]', color='dodgerblue', line_y=line_y) # em
    plot_line(3933.66,'',alpha=0.1, line_y=line_y)
    plot_line(3968.47,'CaII-H,K,H5,[NeIII]', line_y=line_y)
    plot_line(4101.74,r'H$\delta$', color='firebrick', line_y=line_y)
    plot_line(4340.46,r'H$\gamma$', color='firebrick', line_y=line_y)
    plot_line(4861.33,r'H$\beta$', color='firebrick', line_y=line_y)
    plot_line(4958.91,'',alpha=0.1, color='dodgerblue', line_y=line_y)
    plot_line(5006.8,'[OIII]', color='dodgerblue', line_y=line_y) #stronger

    for i, ii in enumerate(np.array([0,2,3,1])):
        plt.axhline(0 + plt_offset*i,lw=3,color='k',alpha=0.3)
#         plt.plot(case1.lam[0], (temps[ii] - np.nanmedian(np.array(temps),0)) + plt_offset*i, color=case_colors[ii])
        plt.plot(case1.lam[0], -(temps[ii] - np.log10(spec_normed)) + plt_offset*i, color=case_colors[ii])

    # plt.xscale('log');
    plt.xlim(5e2,3e5); plt.ylim(llim, ulim*plt_offset)
    plt.xlabel('rest-frame wavelength [$\AA$]')
    # plt.ylabel('$\Delta$ Luminosity [dex] + offset')
    plt.yticks([])
    plt.xlim(3600,5150)
    plt.minorticks_off()
    plt.xticks(np.arange(3700,5250,400), np.round(np.arange(3700,5250,400),0))
    plt.title(r'rest-optical & $4000~\mathrm{\AA}$ break')

    ax4 = fig.add_subplot(gs[0,4])

    plot_line(5876,'HeI', line_y=line_y)
    plot_line(6300,'[OI]', color='dodgerblue', line_y=line_y)
    plot_line(6548,'',alpha=0.1, line_y=line_y)
    plot_line(6562.82,r'H$\alpha$, [NII]', color='firebrick', line_y=line_y)
    plot_line(6584.,'',alpha=0.1, line_y=line_y)
    plot_line(6678,'HeI', line_y=line_y)
    plot_line(6717.0,'',alpha=0.1, line_y=line_y)
    plot_line(6731.3,'[SII]', line_y=line_y)

    for i, ii in enumerate(np.array([0,2,3,1])):
        plt.axhline(0 + plt_offset*i,lw=3,color='k',alpha=0.3)
#         plt.plot(case1.lam[0], (temps[ii] - np.nanmedian(np.array(temps),0)) + plt_offset*i, color=case_colors[ii])
        plt.plot(case1.lam[0], -(temps[ii] - np.log10(spec_normed)) + plt_offset*i, color=case_colors[ii])

    # plt.xscale('log');
    plt.xlim(5e2,3e5); plt.ylim(llim, ulim*plt_offset)
    plt.xlabel('rest-frame wavelength [$\AA$]')
    # plt.ylabel('$\Delta$ Luminosity [dex] + offset')
    plt.yticks([])

    plt.xlim(5810,6800)
    plt.minorticks_off()
    plt.xticks(np.arange(5900,6800,200), np.round(np.arange(5900,6800,200),0))
    plt.title(r'rest-optical & H$\alpha$')
    # plt.tight_layout()

    plt.savefig('figures/spectral_features_v2_'+save_str+'.png',bbox_inches='tight')
#     plt.savefig('figures/spectral_features_v2_fixed.png',bbox_inches='tight')

    plt.show()
    
    
def plot_line(line_wav, line_text, line_y = 0.83, color='k', alpha=0.3, offset=1.0, lw=2.1):
    plt.axvline(line_wav, color=color, alpha=alpha,lw = lw,linestyle='--')
    plt.text(line_wav*offset, line_y, line_text, rotation=270, va = 'top', alpha=alpha+0.2, color=color, fontsize=12)
    
