__author__ = "Soheil Kolouri"
import numpy as np
import cdt as CDT
from scipy import interp
from skimage.transform import radon, iradon

class RadonCDT:
    def __init__(self,theta=np.arange(180)):
        self.theta=theta

    def transform(self,I):
        """
        transform calculates calculates the Radon transform of the input image
        and calculate its CDT with respect to the Radon transform of the template
        input:
            I: A two-dimensional distribution
        output:
            The Radon-CDT transformation of I
        """
        assert not((1.0*I<0).sum())
        # Force I to be a positive probability distribution
        eps=1e-5#This small dc level is needed for a numerically unique solution of the transport map
        I=I+eps
        I=I/I.sum()
        radonI=radon(I,theta=self.theta,circle=False)
        self.dim=radonI.shape[0]
        Ihat=np.zeros_like(radonI)
        self.cdt=CDT.CDT(dim=self.dim)
        for i in range(len(self.theta)):
            Ihat[:,i]=self.cdt.transform(radonI[:,i])
        return Ihat

    def itransform(self,Ihat):
        """
        itransform calculates the inverse of the Radon-CDT. It receives the Radon-CDT
        and finds the corresponding two-dimensional distribution I from it.
        input:
            Ihat: Radon-CDT of I
        output:
            I: The original distribution
        """
        radonI=np.zeros_like(Ihat)
        for i in range(len(self.theta)):
            radonI[:,i]=self.cdt.itransform(Ihat[:,i])
        I=iradon(radonI,theta=self.theta,circle=False)
        return I
