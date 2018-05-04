import numpy as np
from scipy.integrate import cumtrapz
import astropy.constants as c
import astropy.units as u

pc = c.pc.cgs.value
c_light = c.c.cgs.value
a_0 = 1e-5
arcsec_sq = (u.arcsec**2).to(u.sr)  # arcsec**2 in steradian


def planck_B_nu(freq, T):
    """
    Calculates the value of the Planck-Spectrum
    B(nu,T) of a given frequency nu and temperature T

    Arguments
    ---------
    nu : float or array
        frequency in 1/s or with astropy.units

    T: float
        temperature in K or in astropy.units

    Returns:
    --------
    B : float
        value of the Planck-Spectrum at frequency nu and temperature T
        units are using astropy.units if the input values use those, otherwise
        cgs units: erg/(s*sr*cm**2*Hz)

    """
    import numpy as np
    from astropy import units as u
    from astropy import constants as c

    if isinstance(T, u.quantity.Quantity):
        use_units = True
    else:
        T = T * u.K
        use_units = False

    if not isinstance(freq, u.quantity.Quantity):
        freq *= u.Hz

    T = np.array(T.value, ndmin=1) * T.unit
    freq = np.array(freq.value, ndmin=1) * freq.unit

    f_ov_T = freq[np.newaxis, :] / T[:, np.newaxis]
    mx = np.floor(np.log(np.finfo(f_ov_T.ravel()[0].value).max))
    exp = np.minimum(f_ov_T * c.h / c.k_B, mx)
    exp = np.maximum(exp, -mx)

    output = 2 * c.h * freq**3 / c.c**2 / (np.exp(exp) - 1.0) / u.sr

    cgsunit = 'erg/(s*sr*cm**2*Hz)'
    if use_units:
        return output.to(cgsunit).squeeze()
    else:
        return output.to(cgsunit).value.squeeze()


def get_results(r, time, a, sig_g, sig_d, alpha, rho_s, M_star, v_frag, Temp, it, lam, a_opac, k_a, distance=140 * pc, flux_fraction=0.68):

    # interpolate opacity on the same particle size grid as the size distribution

    kappa = np.array([10.**np.interp(np.log10(a), np.log10(a_opac), np.log10(k)) for k in k_a.T]).T  # interpolates the opacity

    if Temp.ndim == 1:
        T = Temp
    else:
        T = Temp[it]

    # reconstruct the size distribution

    sig_da, a_max = get_distri(it, r, a, time, sig_g, sig_d, alpha, rho_s, T, M_star, v_frag, a_0)

    # calculate planck function at every wavelength and radius

    # shape = (n_lam, n_r)
    Bnu = planck_B_nu(c_light / lam, Temp).T

    # calculate optical depth

    # shape = (n_l, n_a, n_r)
    tau = (kappa.T[:, :, np.newaxis] * sig_da[np.newaxis, :, :])
    tau = tau.sum(1)  # shape = (n_l, n_r)

    # calculate intensity at every wavelength and radius for this snapshot
    # here the intensity is still in plain CGS units (per sterad)

    intens = Bnu * (1 - np.exp(-tau))

    # calculate the fluxes

    flux = distance**-2 * cumtrapz(2 * np.pi * r * intens, x=r, axis=1, initial=0)

    # store the integrated flux density in Jy (sanity check: TW Hya @ 870 micron and 54 parsec is about 1.5 Jy)
    flux_t = flux[:, -1] / 1e-23

    # converted intensity to Jy/arcsec**2

    Inu = intens * arcsec_sq / 1e-23

    #   interpolate radius whithin which >=68% of the dust mass is  # Effective radius (see Tripani)

    rf = np.array([np.interp(flux_fraction, _f / _f[-1], r) for _f in flux])

    return rf, flux_t, tau, Inu, sig_da, a_max


def get_distri(it, r, a, time, sig_g, sig_d, alpha, rho_s, Temp, M_star, v_frag, a0=a_0):

    from twopoppy.distribution_reconstruction import reconstruct_size_distribution as rsd

    if Temp.ndim == 1:
        T = Temp
    else:
        T = Temp[it]

    distri = rsd(
        r,
        a,
        time[it],
        sig_g[it],
        sig_d[it],
        alpha,
        rho_s,
        T,
        M_star,
        v_frag,
        a_0=1e-5)
    return distri[:2]
