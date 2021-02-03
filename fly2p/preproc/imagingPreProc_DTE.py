from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import fourier_shift
import numpy as np
import math
import napari
import cv2 as cv

def loadTif(path):
    """ Load in a Scan Image tiff from the specified path
    """
    
    # Make a tiff reader object
    mytiffreader = ScanImageTiffReader(path)
    
    # Get the metadata
    [nCh, discardFBFrames, nDiscardFBFrames, fpv, nVols] = tifMetadata(mytiffreader)
    
    # Load the tif data
    vol = mytiffreader.data()
    
    # Reshape the volume to reflect the experimental parameters
    vol = vol.reshape((int(vol.shape[0]/(fpv*nCh)),fpv,nCh,vol.shape[1], vol.shape[2]))
    
    # Discard the flyback frames
    stack4d = np.squeeze(vol[:,0:fpv-nDiscardFBFrames,0,:,:])
    
    return stack4d


def tifMetadata(mytiffreader):
    """ Load Scan Image tiff metadata from a Scan Image tiff reader object
    """
    
    metadat = mytiffreader.metadata()
    
    # Step through the metadata, extracting relevant parameters
    for i, line in enumerate(metadat.split('\n')):
        if not 'SI.' in line: continue

        # get channel info
        if 'channelSave' in line:
            if not '[' in line:
                nCh = 1
            else:
                nCh = int(line.split('=')[-1].strip())

        if 'scanFrameRate' in line:
            fpsscan = float(line.split('=')[-1].strip())

        if 'discardFlybackFrames' in line:
            discardFBFrames = line.split('=')[-1].strip()

        if 'numDiscardFlybackFrames' in line:
            nDiscardFBFrames = int(line.split('=')[-1].strip())

        if 'numFramesPerVolume' in line:
            fpv = int(line.split('=')[-1].strip())

        if 'numVolumes' in line:
            nVols = int(line.split('=')[-1].strip())
    
    return [nCh, discardFBFrames, nDiscardFBFrames, fpv, nVols]


def tifMotionCorrect(numRefImg, locRefImg, upsampleFactor, stack, sigma):
    """ Motion correct a tiff stack by using phase cross correlation
    numRefImg = the number of images to average for the reference image
    locRefImg = the initial position in the stack to use for the reference
    upsampleFactor = how much to upsample the image in order to shift the image by less than one pixel
    stack = the stack to be registered
    sigma = the sigma to use in Gaussian filtering
    """
    # Generate reference image
    refImg = np.mean(stack[locRefImg:locRefImg+numRefImg,:,:],axis=0)
    
    # Gaussian filter the reference image
    refImgFilt = gaussian_filter(refImg, sigma=sigma)
    
    # Create empty arrays to hold the registration metrics
    shift = np.zeros((2, stack.shape[0]))
    error = np.zeros(stack.shape[0])
    diffphase = np.zeros(stack.shape[0])

    # Create an empty array to hold the motion corrected stack
    stackMC = np.ones(stack.shape).astype('int16')

    # Correct each volume
    for i in range(stack.shape[0]):
        # Get the current image
        shifImg = stack[i,:,:]

        # Filter it
        shifImgFilt = gaussian_filter(shifImg, sigma=sigma)

        # Find the cross correlation between the reference image and the current image
        shift[:,i], error[i], diffphase[i] = phase_cross_correlation(refImgFilt, shifImgFilt, 
                                                                     upsample_factor = upsampleFactor)

        # Shift the image in Fourier space
        offset_image = fourier_shift(np.fft.fftn(shifImg), shift[:,i])
        
        # Convert back and save the motion corrected image
        stackMC[i,:,:] = np.fft.ifftn(offset_image).real.astype('uint16')
        
    return stackMC


def getROIs(stack, roiFN):
    """ Use napari to get ROIs from a stack, using a given ROI function
    """
    
    # Load the mean image in napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(stack.mean(axis=0))
      
    # Use the ROIs that were drawn in napari to get image masks
    allROIs = roiFN(viewer, stack)
    
    return allROIs


def FfromROIs(stack, allROIs):
    """ Calculate the raw fluorescence in each ROI in all ROIS on the given stack
    """
    
    # Initialie the array to hold the fluorescence data
    rawF = np.zeros((stack.shape[0],len(allROIs)))
    
    # Step through each frame in the stack
    for fm in range(0,stack.shape[0]):
        fmNow = stack[fm,:,:]
        
        # Find the sum of the fluorescence in each ROI for the given frame
        for r in range(0,len(allROIs)):
            rawF[fm,r] = np.multiply(fmNow, np.transpose(allROIs[r])).sum()
            
    return rawF


def DFoF(rawF):
    """ Calculate the DF/F given a raw fluorescence signal
    The baseline fluorescence is the mean of the lowest 10% of fluorescence signals
    """
    
    # Initialize the array to hold the DF/F data
    DF = np.zeros(rawF.shape)
    
    # Calculate the DF/F for each ROI
    for r in range(0,rawF.shape[1]):
        Fbaseline = np.sort(rawF[:,r])[0:round(0.1*rawF.shape[0])].mean()
        DF[:,r] = rawF[:,r]/Fbaseline-1
        
    return DF