__author__ = "Soheil Kolouri"
import numpy as np
from scipy import interp

class CDT:
    def __init__(self,dim):
        template=np.ones((dim,))
        template=template/template.sum()
        self.template=template
        self.dim=dim
        self.template_CDF=np.cumsum(template)
        self.x=np.arange(self.dim) #Discretization of the domain of template
        self.xtilde=np.linspace(0,1,self.dim)
        self.template_CDF_inverse=interp(self.xtilde,self.template_CDF,self.x)

    def transform(self,I):
        """
        transform calculates the transport map, f, that morphs the one-dimensional distribution
        I into the template.
        input:
            I: A one dimensional distributions of size self.dim
        output:
            The CDT transformation of I
        """
        assert self.dim==len(I)
        assert not((1.0*I<0).sum())
        # Force I to be a positive probability distribution
        eps=1e-5#This small dc level is needed for a numerically unique solution of the transport map
        I=I+eps
        I=I/I.sum()
        #Calculate its CDF
        I_CDF=np.cumsum(I)
        I_CDF_inverse = interp(self.xtilde,I_CDF, self.x)
        u = interp(self.x,self.template_CDF_inverse,self.template_CDF_inverse-I_CDF_inverse)
        Ihat= u*np.sqrt(self.template)
        return Ihat

    def itransform(self,Ihat):
        """
        itransform calculates the inverse of the CDT. It receives a signal in the CDT space
        and finds the corresponding one dimensional distribution I from it.
        input:
            u: Transport displacement map
            I0: The template used for calculating the CDT
        output:
            I: The original distribution
        """
        u=Ihat/np.sqrt(self.template)
        f=self.x-u
        fprime=np.gradient(f)
        I = interp(self.x,f, self.template/fprime)
        I = I/I.sum()
        return I
