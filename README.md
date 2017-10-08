# Radon Cumulative Distribution Transform


Radon Cumulative distribution transform (Radon-CDT) as described in:

[(1) Kolouri S, Park SR, Rohde GK. The Radon cumulative distribution transform and its application to image classification. IEEE transactions on image processing. 2016 Feb;25(2):920-34.](http://ieeexplore.ieee.org/abstract/document/7358128/)

is a nonlinear and invertible transformation for non-negative d-dimensional signals that guarantees certain linear separation theorems. The transformation builds on top of the Cumulative Distributed Transformation (CDT) described in:

[(2) Park SR, Kolouri S, Kundu S, Rohde GK. The cumulative distribution transform and linear pattern classification. Applied and Computational Harmonic Analysis. 2017 Feb 22.](http://www.sciencedirect.com/science/article/pii/S1063520317300076)

and it extends CDT to d-dimensional probability distributions.  Similar to CDT, Radon-CDT also rises from the rich mathematical foundations of optimal mass transportation and combines it with integral geometry and Radon transform. Unlike the current data extensive nonlinear models, including deep neural networks and their variations, Radon-CDT provides a well-defined invertible nonlinear transformation that could be used alongside linear modeling techniques, including principal component analysis, linear discriminant analysis, and support vector machines (SVM), and does not require extensive training data.

 The corresponding iPython Notebook file for this post could be found [here](https://github.com/skolouri/Radon-Cumulative-Distribution-Transform/blob/master/Radon-CDT_Demo.ipynb). The demo is tested with:

1. numpy '1.13.1'
2. sklearn '0.18.1'
3. skimage '0.13.0'
3. scipy '0.19.1'

Here we first walk you through the formulation of CDT and Radon transform and then introduce the Radon-CDT and demonstrate its applications on various demos.

## Formulation


### Cumulative Distribution Transform (CDT)

Consider two nonnegative one-dimensional signals <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/> defined on <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/82289f06c71b94420b624654501ad06e.svg?invert_in_darkmode" align=middle width=68.09187pt height=22.56408pt/>. Without the loss of generality assume that these signals are normalized so that they could be treated as probability density functions (PDFs). Considering <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> to be a pre-determined 'template'/'reference' PDF, and following the definition of the **optimal mass transportation**  for one-dimensional distributions, one can define the optimal transport map, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/a4fe1ef6edd893e70831c6cf216f5ec3.svg?invert_in_darkmode" align=middle width=76.982895pt height=22.74591pt/> using,

<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/0cbe82a78bb845208169052fcd22e3d1.svg?invert_in_darkmode" align=middle width=232.65165pt height=44.75196pt/></p>

which uniquely associates <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/7211ec64117b386a4d281f03e816f84c.svg?invert_in_darkmode" align=middle width=76.982895pt height=22.74591pt/> to the given density <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/>.

###### Forward Transform:
We use this relationship to define the ** Cumulative Distribution Transform (CDT)** of <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/> (denoted as <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/009c5c6e872cda936558aa1391e2980a.svg?invert_in_darkmode" align=middle width=74.39223pt height=31.0563pt/>), with respect to the reference <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/>:

<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/1efb47898a7c14b03e4b8d53cc237304.svg?invert_in_darkmode" align=middle width=187.26015pt height=19.668165pt/></p>

For one-dimensional PDFs the transport map <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> is uniquely defined, and can be calculated from:

<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/f568e3bda301c41784ffdc1e1bda6086.svg?invert_in_darkmode" align=middle width=137.90568pt height=18.269295pt/></p>

where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/d5678db6a0e35236c7037b64736ccf19.svg?invert_in_darkmode" align=middle width=103.26822pt height=24.56553pt/> and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/066ab7381e7d650225f1730bd6a691a7.svg?invert_in_darkmode" align=middle width=95.777715pt height=24.56553pt/> are the corresponding cumulative distribution functions (CDFs) for <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/3ee98a0ddf705fc4e453f42e3e2563c6.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/>, that is: <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/002d1c8d6f7bd6452b21706524c5b673.svg?invert_in_darkmode" align=middle width=163.540245pt height=28.2282pt/>, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/88ba2440e8b83e1f8a6db6af4599ea63.svg?invert_in_darkmode" align=middle width=149.80515pt height=28.2282pt/>. For continuous positive PDFs <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/>, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> is a continuous and monotonically increasing function. If <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> is differentiable, we can rewrite the above equation as:

<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/2a2097abc9026c829b64577c4de06970.svg?invert_in_darkmode" align=middle width=152.698425pt height=17.250255pt/></p>

###### Inverse Transform

The Inverse-CDT of <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/bb1509c53ed9e94118cb98cd9436ad7f.svg?invert_in_darkmode" align=middle width=10.163505pt height=31.0563pt/> is defined as:
<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/bfd87e6c1d920e5baa94174831278a70.svg?invert_in_darkmode" align=middle width=293.6109pt height=36.953895pt/></p>

where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/86035f674998337e99cf3bd753ab438f.svg?invert_in_darkmode" align=middle width=94.64004pt height=26.70657pt/> refers to the inverse of <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> (i.e. <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/37634a69c6e56ca1ec7efe3f61465c07.svg?invert_in_darkmode" align=middle width=103.288185pt height=26.70657pt/>), and where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/db2e04fa41ca5d29f4ff03898514dda5.svg?invert_in_darkmode" align=middle width=181.347045pt height=31.0563pt/>. The equation above holds for points where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/33799619e6a8adb0933941909e268d50.svg?invert_in_darkmode" align=middle width=15.60933pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> are differentiable. By the construction above, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.780705pt height=22.74591pt/> will be differentiable except for points where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/d906cd9791e4b48a3b848558acda5899.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> are discontinuous. Now we are ready to delve into some exciting applications of CDT.


### Radon Transform

The classic Radon transform, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/0ea3cec9cd8f8324b87218b762b9a686.svg?invert_in_darkmode" align=middle width=13.879635pt height=22.38192pt/>, maps a function  <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/f1ade1a142928a894aa5d160fa999a1d.svg?invert_in_darkmode" align=middle width=79.289595pt height=27.85299pt/> where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/aee9e74580673a1b86cb705eb144212b.svg?invert_in_darkmode" align=middle width=302.188095pt height=27.85299pt/> to the infinite set of its integrals over the hyperplanes of <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/435f1061aa6f25938c3c3515c083d06c.svg?invert_in_darkmode" align=middle width=18.64533pt height=27.85299pt/> and is defined as,
<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/fd4f10308a1196420be12f4ad2f946f2.svg?invert_in_darkmode" align=middle width=239.75325pt height=37.35204pt/></p>

where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/f4793050b7ef4719465234bb92574613.svg?invert_in_darkmode" align=middle width=25.18758pt height=24.56553pt/> is the one-dimensional Dirac delta function.
For <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/12a67427b33187974d4d8bfde354df04.svg?invert_in_darkmode" align=middle width=69.97287pt height=27.85299pt/> where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/6406ab477521e17d339e0878746756d7.svg?invert_in_darkmode" align=middle width=32.67957pt height=27.85299pt/> is the unit sphere in <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/1a7104381f6205cca28decaf37d42180.svg?invert_in_darkmode" align=middle width=18.64533pt height=27.85299pt/>, and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/bc2c22455f069a210505b17f5569628a.svg?invert_in_darkmode" align=middle width=46.89036pt height=22.74591pt/>. Note that <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/5ad8f762edd6de77299fd51a51b5931c.svg?invert_in_darkmode" align=middle width=200.539845pt height=27.85299pt/> in other words the d-dimensional density <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/> is projected into an infinite set of one-dimensional densities, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/3399c52a5d172ff8ac15b73b689363b8.svg?invert_in_darkmode" align=middle width=55.08888pt height=24.56553pt/>. The figure below visualizes the Radon transform for a 2D Gaussian density.

![<img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/45de514e4bd2f5ba36f09fff6b549760.svg?invert_in_darkmode" align=middle width=11.832645pt height=22.38192pt/>s and their corresponding CDT](Figures/figure1.png)

Using the Slice Fourier theorem it can be easily shown that the Radon transform is invertible. The inverse of the Radon transform denoted by <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/0d07777bf454d9f941c8facaea255a08.svg?invert_in_darkmode" align=middle width=30.643305pt height=26.70657pt/> is defined as:
<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/faf70e715418eacd4c2142bd5dba79e4.svg?invert_in_darkmode" align=middle width=453.62955pt height=37.35204pt/></p>
where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/97d6b6abab7046031f875d5a2388a7fd.svg?invert_in_darkmode" align=middle width=26.722575pt height=24.56553pt/>is a one-dimensional high-pass filter with corresponding Fourier transform <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/0bdfe8bd3cb6b25542d577159dc3de62.svg?invert_in_darkmode" align=middle width=118.780695pt height=27.85299pt/> (it appears due to the change of coordinates from spherical to Cartesian in the Fourier Slice theorem) and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/7c74eeb32158ff7c4f67d191b95450fb.svg?invert_in_darkmode" align=middle width=8.188554pt height=15.23973pt/> denotes convolution. The above definition of the inverse Radon transform is also known as the filtered back-projection method, which is extensively used in image reconstruction in the biomedical imaging community. Intuitively each one-dimensional projection/slice, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/3399c52a5d172ff8ac15b73b689363b8.svg?invert_in_darkmode" align=middle width=55.08888pt height=24.56553pt/>, is first filtered via a high-pass filter and then smeared back into <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/1a7104381f6205cca28decaf37d42180.svg?invert_in_darkmode" align=middle width=18.64533pt height=27.85299pt/> along the integration hyperplanes, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/ed7a4a2f4460dd0f7144c5861908c90d.svg?invert_in_darkmode" align=middle width=57.143295pt height=22.74591pt/>, to approximate <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/>. The summation of all smeared approximations then reconstruct <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/>. Finally, we note that Radon transform is a linear transformation.

### Radon-CDT

The idea behind Radon-CDT is to first slice a d-dimensional probability density <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.4843pt height=22.38192pt/> into a set of one-dimensional distributions and then apply CDT to the one dimensional distributions. Formally, given a template distribution <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> the Radon-CDT is defined as:

<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/1358cabdf0d481c1d3136e0f96ae0d66.svg?invert_in_darkmode" align=middle width=376.5597pt height=19.668165pt/></p>

where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/fb6b76d3323483946ea0f3a86521b152.svg?invert_in_darkmode" align=middle width=43.87251pt height=24.56553pt/> satisfies:

<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/4ceae8eec7ac325cd73b832d1af6bf22.svg?invert_in_darkmode" align=middle width=351.69585pt height=43.25079pt/></p>

and the inverse transform is defined as:

<p align="center"><img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/43d06e17c85890d79a4e36e81fab8328.svg?invert_in_darkmode" align=middle width=268.9797pt height=39.30498pt/></p>

where <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/fb6b76d3323483946ea0f3a86521b152.svg?invert_in_darkmode" align=middle width=43.87251pt height=24.56553pt/> can be calculated from <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/bb1509c53ed9e94118cb98cd9436ad7f.svg?invert_in_darkmode" align=middle width=10.163505pt height=31.0563pt/> and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/88fbd05154e7d6a65883f20e1b18a817.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> as  <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/088d8e2edfd62b3a9c94103b7752332b.svg?invert_in_darkmode" align=middle width=158.633145pt height=37.68435pt/>.

## Radon-CDT Demo

Throughout the experiments in this tutorial we assume that <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/b8869cc6e30ce71a54d899877ae2d165.svg?invert_in_darkmode" align=middle width=61.17342pt height=24.56553pt/> is a uniform one-dimensional distribution. Lets start by showing the nonlinear nature of Radon-CDT.

### Nonlinearity

Let <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/d906cd9791e4b48a3b848558acda5899.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> and <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/9eff113852463b85a970d2d65d52280c.svg?invert_in_darkmode" align=middle width=13.727175pt height=22.38192pt/> be the following two images:

![](Figures/I0.bmp) ![](Figures/I1.bmp)

where the images are normalized (sum to one) to be considered as two-dimensional probability distributions. Let <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/a10ff06f53725bca54c49936f91f5fa5.svg?invert_in_darkmode" align=middle width=11.832645pt height=31.0563pt/> denote the corresponding Radon-CDTs. We use the following code to calculate the Radon transform and the Radon-CDT of these images (i.e. distributions).

```python
import numpy as np
from skimage.io import imread
from skimage.transform import radon, iradon
from skimage.color import rgb2gray
import transportBasedTransforms.radonCDT as RCDT
#Get a Radon-CDT object
rcdt=RCDT.RadonCDT()
# Load images and calculate their Radon and Radon-CDT
I=[]
Ir=[]
Ihat=[]
for i in range(2):
    I.append(rgb2gray(imread('./Data/I%d.bmp'%(i))))
    Ir.append(radon(I[i]))
    Ihat.append(rcdt.transform(I[i]))
```
which results in,

![<img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/45de514e4bd2f5ba36f09fff6b549760.svg?invert_in_darkmode" align=middle width=11.832645pt height=22.38192pt/>s and their corresponding Radon-CDT](Figures/figure2.png)

Now to demonstrate the nonlinear nature of Radon-CDT, we choose the simplest linear operator, which is averaging the two distributions. We average the signals in the signal space, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/8a44d333fd9307aaca485044a927c088.svg?invert_in_darkmode" align=middle width=119.29764pt height=24.56553pt/>, in the Radon space <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/defdb60623c584a375721b4996b31a26.svg?invert_in_darkmode" align=middle width=191.322945pt height=26.70657pt/>, and in the Radon-CDT space, <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/5e257929229ad4fbc8fd8107a6cb33e4.svg?invert_in_darkmode" align=middle width=184.600845pt height=31.0563pt/>, and compare the results below.

```python
I3=np.asarray(I).mean(axis=0)
I4=iradon(np.asarray(rI).mean(axis=0))
I5=rcdt.itransform(np.asarray(Ihat).mean(axis=0)
```
which results in,

![Averaging in the signal domain versus in the Radon-CDT domain](Figures/figure3.png)

It can be clearly seen that Radon-CDT provides a nonlinear averaging for these signals.

### Linear separability

Park et al. (2) showed that CDT can turn certain not linearly separable classes of one-dimensional signals into linearly separable ones. Kolouri et al. (1) extended the CDT results to higher dimensional densities and showed that the same characteristics hold for Radon-CDT.

Here we run a toy example to demonstrate this characteristic. We start by defining three classes of two-dimensional images, where Class <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.041505pt height=22.74591pt/>, for <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/1cc5019c20e25f4af278b10609f0574a.svg?invert_in_darkmode" align=middle width=84.613815pt height=24.56553pt/>, consists of translated versions of a <img src="https://rawgit.com/skolouri/Radon-Cumulative-Distribution-Transform/master/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.041505pt height=22.74591pt/>-modal Gaussian distribution. Here we generate these signal classes and their corresponding Radon-CDTs.

```python
N=1000
I=np.zeros((3,N,128,128))
K,L=rcdt.transform(I[0,0,:,:]).shape
Ihat=np.zeros((3,N,K,L))
for c in range(3):
    for i in range(N):
        for _ in range(c+1):
            x,y=np.random.uniform(30,98,(2,)).astype('int')        
            I[c,i,x,y]=1
        I[c,i,:,:]=I[c,i,:,:]/I[c,i,:,:].sum()
        I[c,i,:,:]=filters.gaussian_filter(I[c,i,:,:],sigma=3)    
        Ihat[c,i,:,:]=rcdt.transform(I[c,i,:,:])    
```

Next we run a simple linear classification on these signals in the original space and in the CDT space.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=2) #Get the classifier object
X=np.reshape(I,(3*N,128*128))
Xhat=np.reshape(Ihat,(3*N,K*L))
data=[X,Xhat]
label=np.concatenate((np.zeros(N,),np.ones(N,),-1*np.ones(N,))) # Define the labels as -1,0,1 for the three classes
dataLDA=[[],[]]
for i in range(2):
    dataLDA[i]=lda.fit_transform(data[i],label)
```

Below we visualize the two-dimensional discriminant subspace calculated by the linear discriminant analysis (LDA).

![Visualization of the LDA subspace calculated from the original space and the CDT space.](Figures/figure4.png)

It can be clearly seen that while the classes are not linearly separable in the original space, the CDT representations of the signals is linearly separable.
