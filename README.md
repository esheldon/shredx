# shredx
Run the shredder image deblender on images processed by Source Extractor

## Examples
```python
import shredx

# to process all fof groups at once
loader = shredx.Loader(
     image_files=image_files,
     psf_files=psf_files,
     seg_file=seg_file,
     cat_file=cat_file,
)

results = shredx.shred_fofs(loader)

# to deblend and see some plots as you go
shredx.shred_fofs(loader, show=True)
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
