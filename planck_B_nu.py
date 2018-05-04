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
