from astropy.io import fits
from astropy.wcs import WCS

# Path to your FITS file
fits_path = "P0765000401EPX0003COLIM8000.fits"

# Load FITS data and header
with fits.open(fits_path) as hdul:
    header = hdul[0].header
    data = hdul[0].data

# Reduce to <= 3D
while data.ndim > 3:
    data = data[0]

# Determine spatial shape
if data.ndim == 3:
    nz, ny, nx = data.shape
    z_index = 0  # use first channel
elif data.ndim == 2:
    ny, nx = data.shape
    z_index = None
else:
    raise ValueError("Expected 2D or 3D FITS data.")

# Setup WCS
w = WCS(header)

# Compute center pixel indices
x = nx // 2
y = ny // 2

# Get world coordinates at center
if z_index is None:
    ra_deg, dec_deg = w.pixel_to_world_values(x, y)
else:
    ra_deg, dec_deg, _ = w.pixel_to_world_values(x, y, z_index)

# Print result
print(f"Center RA  (deg): {ra_deg:.6f}")
print(f"Center Dec (deg): {dec_deg:.6f}")

