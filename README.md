# shredx
Run the shredder image deblender on images processed by Source Extractor

## Examples

An example using the library
```python
import shredx

# to process all fof groups at once
loader = shredx.Loader(
     image_files=image_files,
     psf_files=psf_files,
     seg_file=seg_file,
     cat_file=cat_file,
)


# a minimal config, leaving everything else at the defaults
shred_config = {
    'shredding': {
        'psf_ngauss': 5,
    }
}
results = shredx.shred_fofs(loader=loader, shredconf=shred_config)

# to deblend and see some plots as you go
shredx.shred_fofs(
    loader=loader,
    shredconf=shred_config
    show=True,
)
```


Example using the script.  First generate fof groups using fofx
```bash
seg=seg/DES0109+0500_r4969p01_r_segmap.fits
output=fofs/run11-DES0109+0500-fofs.fits

shredx-make-fofs --seg $seg --output $output
```

Then run the shredder using shredx
```bash
config=run11.yaml
seg=seg/DES0109+0500_r4969p01_r_segmap.fits
cat=cat/DES0109+0500_r4969p01_r_cat.fits
fofs=fofs/run11-DES0109+0500-fofs.fits

# we need to make this idmap as we have done for the meds maker
# ids=ids/run11-DES0109+0500-idmap.fits

images="coadd/DES0109+0500_r4969p01_g.fits.fz coadd/DES0109+0500_r4969p01_r.fits.fz coadd/DES0109+0500_r4969p01_i.fits.fz coadd/DES0109+0500_r4969p01_z.fits.fz"
psfs="psf/DES0109+0500_r4969p01_g_psfcat.psf psf/DES0109+0500_r4969p01_r_psfcat.psf psf/DES0109+0500_r4969p01_i_psfcat.psf psf/DES0109+0500_r4969p01_z_psfcat.psf"

start=12800
end=12809
seed=407393828

outfile=output/test.fits

# add this argument when id map is made
#    --ids $ids \
shredx \
    --start $start \
    --end $end \
    --config $config \
    --seed $seed \
    --seg $seg \
    --cat $cat \
    --fofs $fofs \
    --images $images \
    --psfs $psfs \
    --outfile $outfile
```

## Requirements

- numpy
- numba
- shredder
- ngmix
- fitsio
- esutil
- psfex
- fofx (for friends of friends group finding)
- sxdes (optional for tests)
- sep (used by sxdes)

## TODO
- implement saving plots rather than showing
- implement saving information for groups not processed, e.g.
  size 1
- maybe implement converting results to array form for saving
  to disk
