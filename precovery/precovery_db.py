import os
import dataclasses
import healpy as hp
import itertools
import logging
import numpy as np
from typing import (
    Iterable,
    Iterator,
    Optional,
    Union
)

from .config import (
    Config,
    DefaultConfig
)
from .frame_db import (
    FrameDB,
    FrameIndex
)
from .healpix_geom import radec_to_healpixel
from .orbit import (
    Orbit,
    PropagationIntegrator
)
from .spherical_geom import haversine_distance_deg

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60

CANDIDATE_K = 15
CANDIDATE_NSIDE = 2**CANDIDATE_K

logging.basicConfig()
logger = logging.getLogger("precovery")

@dataclasses.dataclass
class PrecoveryCandidate:
    mjd_utc: float
    ra_deg: float
    dec_deg: float
    ra_sigma_arcsec: float
    dec_sigma_arcsec: float
    mag: float
    mag_sigma: float
    filter: str
    obscode: str
    exposure_id: str
    observation_id: str
    healpix_id: int
    pred_ra_deg: float
    pred_dec_deg: float
    pred_vra_degpday: float
    pred_vdec_degpday: float
    delta_ra_arcsec: float
    delta_dec_arcsec: float
    distance_arcsec: float
    dataset_id: str

@dataclasses.dataclass
class FrameCandidate:
    mjd_utc: float
    filter: str
    obscode: str
    exposure_id: str
    healpix_id: int
    pred_ra_deg: float
    pred_dec_deg: float
    pred_vra_degpday: float
    pred_vdec_degpday: float
    dataset_id: str

class PrecoveryDatabase:
    def __init__(self, frames: FrameDB):
        self.frames = frames
        self._exposures_by_obscode: dict = {}

    @classmethod
    def from_dir(cls, directory: str, create: bool = False, mode: str = "r"):
        if not os.path.exists(directory):
            if create:
                return cls.create(directory)

        try:
            config = Config.from_json(
                os.path.join(directory, "config.json")
            )
        except FileNotFoundError:
            config = DefaultConfig
            if not create:
                logger.warning("No configuration file found. Adopting configuration defaults.")

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex.open(frame_idx_db, mode=mode)

        data_path = os.path.join(directory, "data")
        frame_db = FrameDB(
            frame_idx, data_path, config.data_file_max_size, config.nside
        )
        return cls(frame_db)

    @classmethod
    def create(
        cls,
        directory: str,
        nside: int = DefaultConfig.nside,
        data_file_max_size: int = DefaultConfig.data_file_max_size,
    ):
        """
        Create a new database on disk in the given directory.
        """
        os.makedirs(directory)

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex.open(frame_idx_db)

        config = Config(
            nside=nside,
            data_file_max_size=data_file_max_size
        )
        config.to_json(os.path.join(directory, "config.json"))

        data_path = os.path.join(directory, "data")
        os.makedirs(data_path)

        frame_db = FrameDB(frame_idx, data_path, data_file_max_size, nside)
        return cls(frame_db)

    def precover(
        self,
        orbit: Orbit,
        tolerance: float = 30 * ARCSEC,
        max_matches: Optional[int] = None,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        nside: int = 256,
        window_size: int = 7,
        include_frame_candidates: bool = False,
    ):
        requested_nside = nside #We take in a requested nside
    #self.frames.healpix_nside = actual nside of the database/the nside the database is indexed at
    

        """
        Find observations which match orbit in the database. Observations are
        searched in descending order by mjd.
        orbit: The orbit to match.
        max_matches: End once this many matches have been found. If None, find
        all matches.
        start_mjd: Only consider observations from after this epoch
        (inclusive). If None, find all.
        end_mjd: Only consider observations from before this epoch (inclusive).
        If None, find all.
        """
        # basically:
        """
        find all windows between start and end of given size
        for each window:
            propagate to window center
            for each unique epoch,obscode in window:
                propagate to epoch
                find frames which match healpix of propagation
                for each matching frame
                    find matching observations
                    for each matching observation
                        yield match
        """
        if start_mjd is None or end_mjd is None:
            first, last = self.frames.idx.mjd_bounds()
            if start_mjd is None:
                start_mjd = first
            if end_mjd is None:
                end_mjd = last

        n = 0
        logger.info(
            "precovering orbit %s from %f.6f to %f.5f, window=%d",
            orbit.orbit_id,
            start_mjd,
            end_mjd,
            window_size,
        )

        windows = self.frames.idx.window_centers(start_mjd, end_mjd, window_size)

        # group windows by obscodes so that many windows can be searched at once
        for obscode, obs_windows in itertools.groupby(
            windows, key=lambda pair: pair[1]
        ):
            mjds = [window[0] for window in obs_windows]


            #We now check for matches at a requested nside
            matches_requested_nside = self._check_windows(
                requested_nside,
                mjds,
                obscode,
                orbit,
                tolerance,
                start_mjd=start_mjd,
                end_mjd=end_mjd,
                window_size=window_size,
                include_frame_candidates=include_frame_candidates,
            )
            
            for result in matches_requested_nside:
                yield result
                n += 1
                if max_matches is not None and n >= max_matches:
                    return
    
    def _check_windows(
        self,
        requested_nside,
        window_midpoints: Iterable[float],
        obscode: str,
        orbit: Orbit,
        tolerance: float,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        include_frame_candidates: bool = False,
    ):
        """
        Find all observations that match orbit within a list of windows
        """

        #Used Latter for resampling healpixels from requested_nside to actual_nside
        nside_ratio = int(self.frames.healpix_nside/requested_nside)
        bitwise_factor = 2*(self.findPosition(nside_ratio)-1)



        # Propagate the orbit with n-body to every window center
        orbit_propagated = orbit.propagate(window_midpoints, PropagationIntegrator.N_BODY)

        # Calculate the location of the orbit on the sky with n-body propagation
        window_ephems = orbit.compute_ephemeris(obscode, window_midpoints, PropagationIntegrator.N_BODY)
    


        #The window_healpixels calculates healpixels using the given requested_nside. We now only check windows found using the requested_nside.
        window_healpixels = radec_to_healpixel(
            np.array([w.ra for w in window_ephems]),
            np.array([w.dec for w in window_ephems]),
            requested_nside,
            bitwise_factor = 0
        ).astype(int)

        # Using the propagated orbits, check each window. Propagate the orbit from the center of
        # window using 2-body to find any HealpixFrames where a detection could have occured
        for window_midpoint, window_ephem, window_healpixel, orbit_window in zip(
            window_midpoints, window_ephems, window_healpixels, orbit_propagated
        ):
            start_mjd_window = window_midpoint - (window_size / 2)
            end_mjd_window = window_midpoint + (window_size / 2)

            # Check if start_mjd_window is not earlier than start_mjd (if defined)
            # If start_mjd_window is earlier, then set start_mjd_window to start_mjd
            if (start_mjd is not None) and (start_mjd_window < start_mjd):
                logger.info(f"Window start MJD [UTC] ({start_mjd_window}) is earlier than desired start MJD [UTC] ({start_mjd}).")
                start_mjd_window = start_mjd

            # Check if end_mjd_window is not later than end_mjd (if defined)
            # If end_mjd_window is later, then set end_mjd_window to end_mjd
            if (end_mjd is not None) and (end_mjd_window > end_mjd):
                logger.info(f"Window end MJD [UTC] ({end_mjd_window}) is later than desired end MJD [UTC] ({end_mjd}).")
                end_mjd_window = end_mjd

            timedeltas = []
            test_mjds = []
            test_healpixels = []
            for mjd, healpixels in self.frames.idx.propagation_targets(
                start_mjd_window, end_mjd_window, obscode
            ):
                logger.debug("mjd=%.6f:\thealpixels with data: %r", mjd, healpixels)
                timedelta = mjd - window_midpoint
                timedeltas.append(timedelta)
                test_mjds.append(mjd)
                test_healpixels.append(healpixels)

            approx_ras, approx_decs = window_ephem.approximately_propagate(
                obscode,
                orbit_window,
                timedeltas,
            )

            #approx_healpixels is run at requested nside. radec_to_healpixel returns the resampled_healpixel
            approx_healpixels = radec_to_healpixel(
                approx_ras, approx_decs, requested_nside, bitwise_factor
                #radec_to_healpixel will return hp.ang2pix(nside, ra, dec, nest=True, lonlat=True).astype(np.int64). We currently have the downsampled_healpixel
            ).astype(int) 

            #We generate a 2D array of resampled healpixels
            resampled_healpixels_list= []
            for healpixel in approx_healpixels:
                resampled_healpixels_list.append(list(range(healpixel, healpixel+ 2**(bitwise_factor))))
            resampled_healpixels = np.array(resampled_healpixels_list)


            keep_mjds = []
            keep_approx_healpixels = []

            for mjd, healpixels, approx_ra, approx_dec, resampled_healpixel in zip(
                test_mjds, test_healpixels, approx_ras, approx_decs, resampled_healpixels
            ):

                for healpixel in resampled_healpixel:
                    logger.debug("mjd=%.6f:\thealpixels with data: %r", mjd, healpixels)
                    logger.debug(
                        "mjd=%.6f:\tephemeris at ra=%.3f\tdec=%.3f\thealpix=%d",
                        mjd,
                        approx_ra,
                        approx_dec,
                        healpixel,
                    )
            
                    if healpixel not in healpixels:
                        # No exposures anywhere near the ephem, so move on.
                        continue
                    logger.debug("mjd=%.6f: healpixel collision, checking frames", mjd)
                    keep_mjds.append(mjd)
                    keep_approx_healpixels.append(healpixel)
                
            if len(keep_mjds) > 0:
                matches = self._check_frames(
                    orbit_window,
                    keep_approx_healpixels,
                    obscode,                        
                    keep_mjds,                    
                    tolerance,
                    include_frame_candidates
                )
                for m in matches:
                    yield m
            #[m for m in self._check_frames(orbit_window, keep_approx_healpixels, obscode, keep_mjds, tolerance, include_frame_candidates)]


    
    #Helper function returns true if a number is not a power of 2.
    def isPowerofTwo(self, n):
        return (True if(n > 0 and ((n & (n - 1)) > 0)) else False)

    #Helper function that finds the position of the bit set to 1
    def findPosition(self, n):
        if (self.isPowerofTwo(n) == True):
            return -1
        i = 1
        pos = 1
        while ((i & n) == 0):
            i = i << 1
            pos += 1
        return pos
 


    def _check_frames(
        self,
        orbit: Orbit,
        healpixels: Iterable[int],
        obscode: str,
        mjds: Iterable[float],
        tolerance: float,
        include_frame_candidates: bool
    ) -> Iterator[Union[PrecoveryCandidate, FrameCandidate]]:
        """
        Deeply inspect all frames that match the given obscode, mjd, and healpix to
        see if they contain observations which match the ephemeris.
        """
        # Compute the position of the ephem carefully.
        exact_ephems = orbit.compute_ephemeris(obscode, mjds)
        for exact_ephem, mjd, healpix in zip(exact_ephems, mjds, healpixels):
            frames = self.frames.idx.get_frames(obscode, mjd, healpix)
            logger.info(
                "checking frames for healpix=%d obscode=%s mjd=%f",
                healpix,
                obscode,
                mjd,
            )
            n_frame = 0

            # Calculate the HEALpixel ID for the predicted ephemeris
            # of the orbit with a high nside value (k=15, nside=2**15)
            # The indexed observations are indexed to a much lower nside but
            # we may decide in the future to re-index the database using different
            # values for that parameter. As long as we return a Healpix ID generated with
            # nside greater than the indexed database then we can always down-sample the
            # ID to a lower nside value
            healpix_id = int(radec_to_healpixel(
                exact_ephem.ra,
                exact_ephem.dec,
                nside=CANDIDATE_NSIDE,
                bitwise_factor = 0
            ))
            #bitwise_factor = 0 ensures that there are no bitwise operations (bitwise operations are later needed for resampling)

            for f in frames:
                logger.info("checking frame: %s", f)
                obs = np.array(list(self.frames.iterate_observations(f)))
                n = len(obs)
                obs_ras = np.array([o.ra for o in obs])
                obs_decs = np.array([o.dec for o in obs])
                distances = haversine_distance_deg(
                    exact_ephem.ra,
                    obs_ras,
                    exact_ephem.dec,
                    obs_decs,
                )
                dras = exact_ephem.ra - obs_ras
                ddecs = exact_ephem.dec - obs_decs
                # filter to observations with distance below tolerance
                idx = distances < tolerance
                distances = distances[idx]
                dras = dras[idx]
                ddecs = ddecs[idx]
                obs = obs[idx]
                for o, distance, dra, ddec in zip(obs, distances, dras, ddecs):
                    candidate = PrecoveryCandidate(
                        mjd_utc=f.mjd,
                        ra_deg=o.ra,
                        dec_deg=o.dec,
                        ra_sigma_arcsec=o.ra_sigma/ARCSEC,
                        dec_sigma_arcsec=o.dec_sigma/ARCSEC,
                        mag=o.mag,
                        mag_sigma=o.mag_sigma,
                        filter=f.filter,
                        obscode=f.obscode,
                        exposure_id=f.exposure_id,
                        observation_id=o.id.decode(),
                        healpix_id=healpix_id,
                        pred_ra_deg=exact_ephem.ra,
                        pred_dec_deg=exact_ephem.dec,
                        pred_vra_degpday=exact_ephem.ra_velocity,
                        pred_vdec_degpday=exact_ephem.dec_velocity,
                        delta_ra_arcsec=dra/ARCSEC,
                        delta_dec_arcsec=ddec/ARCSEC,
                        distance_arcsec=distance/ARCSEC,
                        dataset_id=f.dataset_id,
                    )
                    yield candidate

                logger.info("checked %d observations in frame", n)
                if (len(obs) == 0) & (include_frame_candidates):
                    frame_candidate = FrameCandidate(
                        mjd_utc=f.mjd,
                        filter=f.filter,
                        obscode=f.obscode,
                        exposure_id=f.exposure_id,
                        healpix_id=healpix_id,
                        pred_ra_deg=exact_ephem.ra,
                        pred_dec_deg=exact_ephem.dec,
                        pred_vra_degpday=exact_ephem.ra_velocity,
                        pred_vdec_degpday=exact_ephem.dec_velocity,
                        dataset_id=f.dataset_id,
                    )
                    yield frame_candidate

                    logger.info(f"no observations found in this frame")

                n_frame += 1
            logger.info("checked %d frames", n_frame)