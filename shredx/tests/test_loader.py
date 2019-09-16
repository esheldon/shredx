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
import pytest
import shredx
from tempfile import TemporaryDirectory


@pytest.mark.parametrize('seed', [6419, 491])
def test_loader_smoke(seed):
    """
    test that the loader doesn't crash
    """

    shredx.setup_logging('debug')
    rng = np.random.RandomState(seed)

    with TemporaryDirectory() as tmpdir:
        image_files, psfs, seg_file, cat_file = \
            shredx.sim.get_simulated_files(tmpdir, rng=rng)

        loader = shredx.Loader(
            image_files=image_files,
            psf_files=psfs,
            seg_file=seg_file,
            cat_file=cat_file,
            coord_offset=0,
        )

        # or whatever
        numbers = [1, 2, 3]

        mbobs = loader.get_mbobs(numbers)  # noqa
