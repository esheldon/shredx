"""
TODO
    - example psfex file
    - example g,r,i,z image files with weight maps
    - example seg file
    - example cat file
    - we can generate these using the simulation in
      shredder, with a fake psfex file from anywhere.
      will need to fake the wcs header
"""

import numpy as np
import shredder
import shredx
from tempfile import TemporaryDirectory


def _write_simulated_files(tmpdir, seed):
    rng = np.random.RandomState(seed)
    sim = shredder.Sim(rng=rng)

    mbobs = sim()

    # run sep to get catalog and seg map, write
    # to files


def test_loader_smoke(seed):

    with TemporaryDirectory() as tmpdir:
        image_files, seg_file, cat_file = _write_simulated_files(tmpdir, seed)

        nband = len(image_files)
        psf_files = ['something']*nband

        loader = shredx.Loader(
            image_files=image_files,
            psf_files=psf_files,
            seg_file=seg_file,
            cat_file=cat_file,
            coord_offset=0,
        )

        # or whatever
        numbers = [1, 2, 3]

        mbobs = loader.get_mbobs(numbers)
