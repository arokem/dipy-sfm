## `dipy-sfm`

This container calculates SFM parameters [Rokem2015]_, based on diffusion MRI
data.

.. [Rokem2015] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
   N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
   (2015). Evaluating the accuracy of diffusion MRI models in white
   matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272

Parameters
---------

fdata : str

The name of a nifti file with preprocessed diffusion data.

fbval : str

The name of a text-file with b-values in FSL format.

fbvec : str

The name of a text file with the b-vectors in FSL format.

fmask : str, optional

The name of a nifti file containing boolean mask of locations to
analyze. Default: no masking


Metadata
--------
The mounted `input` folder should contain a `metadata.json` file with the following
format:

    {
    "fdata":"HARDI150.nii.gz",
    "fbval":"HARDI150.bval",
    "fbvec":"HARDI150.bvec",
    "fmask":"mask.nii.gz"
    }

Where `fmask` parameter is optional.

Returns
-------
`root_sfm.nii.gz` : file
    A nifti file containing the 362 SFM parameters.

`root_{fa, di}`: files
   Nifti files containing the Fiber Anisotrpopy (FA), and Dispersion Index (DI).

Examples
-------
To run this container use:

    docker run --rm -it -v /path/to/data:/input -v /path/to/output/:/output arokem/dipy-sfm

Where the folder `/path/to/data/` should contain the `metadata.json` file,

Notes
-----
This uses the `dipy.reconst.sfm` module: http://nipy.org/dipy/reference/dipy.reconst.html#module-dipy.reconst.sfm
