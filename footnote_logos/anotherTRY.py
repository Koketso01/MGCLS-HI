#!/usr/bin/env python3
"""
Final pass: Optical (inverted) + smooth, line-only X-ray ICM contours.
- Takes FIRST celestial plane from both FITS.
- Reprojects X-ray to Optical WCS (alignment fixed).
- Strong Gaussian smoothing (sigma≈5.5 px).
- Uses user-specified contour levels (negatives allowed).
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aplpy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning

# --------- INPUTS (your paths) ----------
OPTICAL_PATH = "./J0314.3-4525-cutout-CDS_P_DESI-Legacy-Surveys_DR10_color_2.0deg_5000x5000Pixels_SIN.fits"
XRAY_PATH    = "./P0765000401EPX0003COLIM8000.fits"
OUTPUT_PATH  = "./optical_xray_contours_final.png"

# Your contour stats (exact):
USER_LEVELS = np.array(
    [-0.00129085, -0.00023633, 0.00081819, 0.00187271, 0.00292723],
    dtype=float
)

# “Smoothness level 11” → sigma ≈ 11/2 px
XRAY_SMOOTH_SIGMA_PIX = 5.5

# --------- HELPERS ----------
def open_first_image_hdu(path):
    hdul = fits.open(path, memmap=True)
    # prefer primary if it has data; else first ImageHDU/CompImageHDU with data
    hdu = hdul[0] if hdul[0].data is not None else next(
        (hh for hh in hdul if isinstance(hh, (fits.ImageHDU, fits.CompImageHDU)) and hh.data is not None),
        None
    )
    if hdu is None:
        hdul.close()
        raise ValueError(f"No image data found in {path}")
    return hdu, hdul

def first_celestial_2d(hdu):
    """Select FIRST index for all non-celestial axes; keep last two (celestial)."""
    data = np.asarray(hdu.data)
    if data is None:
        raise ValueError("HDU has no data")
    if data.ndim < 2:
        raise ValueError(f"Need at least 2D, got ndim={data.ndim}")
    if data.ndim == 2:
        return data, WCS(hdu.header).celestial
    # For FITS order, leading axes are non-spatial; lock them to 0:
    sl = tuple([0] * (data.ndim - 2) + [slice(None), slice(None)])
    arr2d = data[sl]
    if arr2d.ndim != 2:
        raise ValueError("Slicing did not yield 2D")
    return arr2d, WCS(hdu.header).celestial

def gaussian_smooth(img, sigma_pix):
    if sigma_pix and sigma_pix > 0:
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        ker = Gaussian2DKernel(sigma_pix)
        return convolve_fft(img, ker, allow_huge=True, boundary="fill", fill_value=np.nan)
    return img

# --------- MAIN ----------
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=AstropyWarning)

    # Load + slice first planes
    opt_hdu, opt_hdul = open_first_image_hdu(OPTICAL_PATH)
    x_hdu,   x_hdul   = open_first_image_hdu(XRAY_PATH)

    optical_2d, optical_wcs = first_celestial_2d(opt_hdu)
    xray_2d,   xray_wcs     = first_celestial_2d(x_hdu)

    # Reproject X-ray onto optical WCS grid (align RA/Dec)
    try:
        from reproject import reproject_interp
    except Exception:
        opt_hdul.close(); x_hdul.close()
        sys.exit("Please install reproject:  pip install reproject")
    target = optical_wcs.to_header()
    target["NAXIS"]  = 2
    target["NAXIS1"] = optical_2d.shape[1]
    target["NAXIS2"] = optical_2d.shape[0]
    xray_reproj, _ = reproject_interp((xray_2d, xray_wcs), target)

    # Keep negatives (since user levels include negatives); only drop non-finite
    x = np.array(xray_reproj, dtype=float)
    x[~np.isfinite(x)] = np.nan

    # Strong smoothing for smooth contour lines
    x = gaussian_smooth(x, XRAY_SMOOTH_SIGMA_PIX)

    # Optical display stretch (inverted)
    opt = np.array(optical_2d, dtype=float)
    finite_opt = np.isfinite(opt)
    vmin = np.nanpercentile(opt[finite_opt], 0.5)
    vmax = np.nanpercentile(opt[finite_opt], 99.5)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    # Levels: clean & sorted (keep negatives)
    levels = np.array(sorted({float(L) for L in USER_LEVELS if np.isfinite(L)}))
    if levels.size == 0:
        opt_hdul.close(); x_hdul.close()
        sys.exit("No valid contour levels.")

    # ---- PLOT ----
    fig = plt.figure(figsize=(11, 9))
    f = aplpy.FITSFigure(optical_2d, figure=fig, wcs=optical_wcs)

    f.show_colorscale(cmap="gray", stretch="asinh", vmin=vmin, vmax=vmax)   #, invert=True)

    # Line-only contours; pass ndarray + wcs=optical_wcs (shared WCS)
    f.show_contour(
        x,
        wcs=optical_wcs,
        levels=levels,
        colors="red",
        linewidths=2.0,
        linestyles="solid",
        filled=False,
        overlap=True,
        antialiased=True
    )

    # Cosmetics
    f.add_grid(); f.grid.set_color("white"); f.grid.set_linestyle(":"); f.grid.set_alpha(0.35)
    #f.ticks.set_color("white"); f.tick_labels.set_color("white"); f.axis_labels.set_color("white")
    f.set_title("Optical (inverted) with clean X-ray ICM contours", color="white", fontsize=13)
    fig.patch.set_facecolor("black")
    f._figure.axes[0].set_facecolor("black")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)

    x_hdul.close(); opt_hdul.close()

print(f"[OK] Saved {OUTPUT_PATH}")
print("Levels:", levels)
print("Smoothing sigma (px):", XRAY_SMOOTH_SIGMA_PIX)


