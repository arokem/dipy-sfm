#!/usr/bin/env python

import os.path as op
import json

import numpy as np
import nibabel as nib
import dipy.reconst.sfm as sfm
import dipy.core.gradients as grad

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
    return np.sum(beta, -1) / iso

def calc_di(beta, mask=None):
    """
    Calculate the Dispersion Index:

    Calculate a dispersion index based on the formula:

    $DI = \frac{\sum_{i=2}^{n}{\beta_i^2 alpha_i}}{\sum{i=1}{n}{\beta_i^2}}$

    where $\beta_i$ is the weight in each direction, denoted by $alpha_i$,
    relative to the direction of the maximal weight.
    """
    di = np.zeros(beta.shape)
    if mask is None:
        mask = np.ones(beta.shape, bool)

    di_flat = np.zeros(np.sum(mask))
    mask_beta = beta[mask]

    for vox in xrange(di_fla.shape[0]):
        inds = np.argsort(mask_beta[vox])[::-1] # From largest to
                                                        # smallest
        nonzero_idx = np.where(mask_beta[vox][inds]>0)
        if len(nonzero_idx[0])>0:
            # Only look at the non-zero weights:
            vox_idx = inds[nonzero_idx].astype(int)
            this_mp = mask_beta[vox][vox_idx]
            # XXX REPLACE WITH THE SPHERE:
            this_dirs = self.rot_vecs.T[vox_idx]
            n_idx = len(vox_idx)
            if all_to_all:
                di_s = np.zeros(n_idx)
                # Calculate this as all-to-all:
                angles = np.arccos(np.dot(this_dirs, this_dirs.T))
                for ii in xrange(n_idx):
                    this_di_s = 0
                    for jj in  xrange(ii+1, n_idx):
                        ang = angles[ii, jj]
                        di_s[ii] += np.sin(ang) * ((this_mp[ii]*this_mp[jj])/
                                           np.sum(this_mp**2))

                di_flat[vox] = np.mean(di_s)/n_idx
            else:

                #Calculate this from the highest peak to each one of the
                #others:
                this_pdd, dirs = this_dirs[0], this_dirs[1:]
                angles = np.arccos(np.dot(dirs, this_pdd))
                angles = np.min(np.vstack([angles, np.pi-angles]), 0)
                angles = angles/(np.pi/2)
                di_flat[vox] = np.dot(this_mp[1:]**2/np.sum(this_mp**2),
                                      np.sin(angles))

    out = ozu.nans(self.signal.shape[:3])
    out[self.mask] = di_flat
    return out



if __name__=="__main__":
    fmetadata = "/input/metadata.json"
    # Fetch metadata:
    metadata = json.load(open(fmetadata))
    fdata, fbval, fbvec = [metadata[k] for k in ["fdata", "fbval", "fbvec"]]

    # Load the data:
    img = nib.load(op.join('/input', str(fdata)))
    gtab = grad.gradient_table(op.join('/input', str(fbval)),
                               op.join('/input', str(fbvec)))
    data = img.get_data()
    affine = img.get_affine()

    # Get the optional params:
    fmask = metadata.get('fmask', None)
    if fmask is None:
        mask = None
    else:
        mask = nib.load(fmask).get_data().astype(bool)

    # Fit the model:
    sfmodel = sfm.SparseFascicleModel(gtab)
    sffit = sfmodel.fit(data, mask=mask)

    # The output will all have the same basic name as the data file-name:
    root = op.join("/output",
                   op.splitext(op.splitext(op.split(fdata)[-1])[0])[0])

    # Save to file:
    dpsave(sffit.beta, root + '_SFM_params.nii.gz')
    sf_fa = calc_fa(sffit.beta, sffit.iso, mask=mask)
    dpsave(sffit.beta, root + '_SFM_FA.nii.gz')
    sf_di = calc_di(sffit.beta, mask=mask)
    dpsave(sffit.beta, root + '_SFM_DI.nii.gz')
