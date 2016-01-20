#!/usr/bin/env python

import os.path as op
import json

import numpy as np
import nibabel as nib
import dipy.reconst.sfm as sfm
import dipy.core.gradients as grad
import dipy.data as dpd

def dpsave(img, filename):
    """
    Some tools require the qform code to be 1. We set the affine, qform,
    and sfrom to be the same for maximum portability.
    """
    affine = img.get_affine()
    img.set_sform(affine, 1)
    img.set_qform(affine, 1)
    nib.save(img, filename)

def calc_fa(beta, iso, mask=None):
    """
    Calculate the Fiber Anistropy in each voxel sum(beta_i)/w_0
    """
    if mask is None:
        mask = np.ones(beta.shape[:-1], bool)

    fa_flat = np.sum(beta, -1)[mask] / iso.predict()[:, 0]

    fa = np.zeros(beta.shape[:-1])
    fa[mask] = fa_flat
    return fa

def calc_di(beta, sphere, mask=None):
    """
    Calculate the Dispersion Index:

    Calculate a dispersion index based on the formula:

    $DI = \frac{\sum_{i=2}^{n}{\beta_i^2 alpha_i}}{\sum{i=1}{n}{\beta_i^2}}$

    where $\beta_i$ is the weight in each direction, denoted by $alpha_i$,
    relative to the direction of the maximal weight.
    """
    di = np.zeros(beta.shape[:-1])
    if mask is None:
        mask = np.ones(beta.shape[:-1], bool)

    di_flat = np.zeros(np.sum(mask))
    mask_beta = beta[mask]

    for vox in xrange(di_flat.shape[0]):
        inds = np.argsort(mask_beta[vox])[::-1]  # From largest to smallest
        nonzero_idx = np.where(mask_beta[vox][inds] > 0)
        if len(nonzero_idx[0]) > 0:
            # Only look at the non-zero weights:
            vox_idx = inds[nonzero_idx].astype(int)
            this_mp = mask_beta[vox][vox_idx]
            this_dirs = sphere.vertices[vox_idx]
            n_idx = len(vox_idx)
            this_pdd, dirs = this_dirs[0], this_dirs[1:]
            angles = np.arccos(np.dot(dirs, this_pdd))
            angles = np.min(np.vstack([angles, np.pi-angles]), 0)
            angles = angles/(np.pi/2)
            di_flat[vox] = np.dot(this_mp[1:]**2/np.sum(this_mp**2),
                                  np.sin(angles))

    di[mask] = di_flat
    return di


if __name__ == "__main__":
    fmetadata = '/input/metadata.json'
    # Fetch metadata:
    with open(fmetadata, 'rt') as fobj:
        metadata = json.load(fobj)
    fdata, fbval, fbvec = [metadata[k] for k in ["fdata", "fbval", "fbvec"]]

    # Load the data:
    img = nib.load(op.join('/input', str(fdata)))
    gtab = grad.gradient_table(op.join('/input', str(fbval)),
                               op.join('/input', str(fbvec)))
    data = img.get_data()
    affine = img.get_affine()

    # Get the optional mask param:
    fmask = metadata.get('fmask', None)
    if fmask is None:
        mask = None
    else:
        mask = nib.load(op.join('/input', fmask)).get_data().astype(bool)

    # Fit the model:
    sfmodel = sfm.SparseFascicleModel(gtab)
    sffit = sfmodel.fit(data, mask=mask)

    # The output will all have the same basic name as the data file-name:
    root = op.join("/output",
                   op.splitext(op.splitext(op.split(fdata)[-1])[0])[0])

    # Save to files:
    dpsave(nib.Nifti1Image(sffit.beta, affine), root + '_SFM_params.nii.gz')
    sf_fa = calc_fa(sffit.beta, sffit.iso, mask=mask)
    dpsave(nib.Nifti1Image(sf_fa, affine), root + '_SFM_FA.nii.gz')
    sphere = dpd.get_sphere()
    sf_di = calc_di(sffit.beta, sphere, mask=mask)
    dpsave(nib.Nifti1Image(sf_di, affine), root + '_SFM_DI.nii.gz')
