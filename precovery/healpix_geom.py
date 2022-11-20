import healpy as hp
import numpy as np
import numpy.typing as npt

def radec_to_healpixel(ra: npt.NDArray[np.float64], dec: npt.NDArray[np.float64], nside: int, bitwise_factor: int) -> npt.NDArray[np.int64]:
    if bitwise_factor == 0:
        return hp.ang2pix(nside, ra, dec, nest=True, lonlat=True).astype(np.int64) 
    else:
        return (hp.ang2pix(nside, ra, dec, nest=True, lonlat=True) << bitwise_factor).astype(np.int64)
    
    ##bitwise operations > 0 means the healpixels will be returned after resampling
