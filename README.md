# Radon Cumulative Distribution Transform


Radon Cumulative distribution transform (Radon-CDT) as described in:

[(1) Kolouri S, Park SR, Rohde GK. The Radon cumulative distribution transform and its application to image classification. IEEE transactions on image processing. 2016 Feb;25(2):920-34.](http://ieeexplore.ieee.org/abstract/document/7358128/)

is a nonlinear and invertible transformation for non-negative d-dimensional signals that guarantees certain linear separation theorems. The transformation builds on top of the Cumulative Distributed Transformation (CDT) described in:

[(2) Park SR, Kolouri S, Kundu S, Rohde GK. The cumulative distribution transform and linear pattern classification. Applied and Computational Harmonic Analysis. 2017 Feb 22.](http://www.sciencedirect.com/science/article/pii/S1063520317300076)

and it extends CDT to d-dimensional probability distributions.  Similar to CDT, Radon-CDT also rises from the rich mathematical foundations of optimal mass transportation and combines it with integral geometry and Radon transform. Unlike the current data extensive nonlinear models, including deep neural networks and their variations, Radon-CDT provides a well-defined invertible nonlinear transformation that could be used alongside linear modeling techniques, including principal component analysis, linear discriminant analysis, and support vector machines (SVM), and does not require extensive training data.

 The corresponding iPython Notebook file for this post could be find [here](https://github.com/skolouri/Radon-Cumulative-Distribution-Transform/blob/master/Radon_CDT_Demo.ipynb). The demo is tested with:

1. numpy '1.13.1'
2. sklearn '0.18.1'
3. skimage '0.13.0'
3. scipy '0.19.1'

Here we first walk you through the formulation of CDT and Radon transform and then introduce the Radon-CDT and demonstrate its applications on various demos.

## Formulation


### Cumulative Distribution Transform (CDT)

Consider two nonnegative one-dimensional signals $I_0$ and $I$ defined on $X,Y\subset\mathbb{R}$. Without the loss of generality assume that these signals are normalized so that they could be treated as probability density functions (PDFs). Considering $I_0$ to be a pre-determined 'template'/'reference' PDF, and following the definition of the **optimal mass transportation**  for one-dimensional distributions, one can define the optimal transport map, $f:X \rightarrow Y$ using,

$$
\int_{inf(Y)}^{f(x)} I(\tau) d\tau=\int_{inf(X)}^{x}I_0(\tau)d\tau
$$

which uniquely associates $f:X\rightarrow Y$ to the given density $I$.

###### Forward Transform:
We use this relationship to define the ** Cumulative Distribution Transform (CDT)** of $I$ (denoted as $\hat{I}: X \to \mathbb{R}$), with respect to the reference $I_0$:

$$
\hat{I}(x) = \left(  f(x) - x \right) \sqrt{I_0(x)}.
$$

For one-dimensional PDFs the transport map $f$ is uniquely defined, and can be calculated from:

$$
f(x)=J^{-1}(J_0(x)).
$$

where $J_0: X \to [0,1]$ and $J: Y \to [0,1]$ are the corresponding cumulative distribution functions (CDFs) for ${I}_0$ and $I$, that is: $J_0(x) = \int_{\inf(X)}^x I_0(\tau) d\tau$, $J(y) = \int_{\inf(Y)}^y I(\tau) d\tau$. For continuous positive PDFs $I_0$ and $I$, $f$ is a continuous and monotonically increasing function. If $f$ is differentiable, we can rewrite the above equation as:

$$
I_0(x) = f^{\prime}(x) I(f(x)).
$$

###### Inverse Transform

The Inverse-CDT of $\hat{I}$ is defined as:
$$
I(y) = \frac{d }{dy}J_0(f_1^{-1}(y)) = (f^{-1})^{\prime} I_0(f^{-1}(y))
$$

where $f^{-1}: Y \to X$ refers to the inverse of $f$ (i.e. $f^{-1}(f(x)) = x$), and where $f(x) =  {\hat{I}_1(x)}/{\sqrt{I_0(x)}} + x$. The equation above holds for points where $J_0$ and $f$ are differentiable. By the construction above, $f$ will be differentiable except for points where $I_0$ and $I_1$ are discontinuous. Now we are ready to delve into some exciting applications of CDT.


### Radon Transform

The classic Radon transform, $\mathcal{R}$, maps a function  $I\in L^1(\mathbb{R}^d)$ where $L^1(\mathbb{R}^d):=\{ I:\mathbb{R}^d \rightarrow \mathbb{R} | \int_{\mathbb{R}^d} |I(x)|dx \leq \infty\}$ to the infinite set of its integrals over the hyperplanes of $\mathbb{R}^d$ and is defined as,
$$
\mathcal{R} I(t,\theta):=\int_{\mathbb{R}^d}I(x)\delta(t-x\cdot\theta)dx,
$$

where $\delta(.)$ is the one-dimensional Dirac delta function.
For $\forall\theta\in \mathbb{S}^{d-1}$ where $\mathbb{S}^{d-1}$ is the unit sphere in $\mathbb{R}^{d}$, and $\forall t \in \mathbb{R}$. Note that $\mathcal{R}: L^1(\mathbb{R}^d)\rightarrow L^1(\mathbb{R}\times \mathbb{S}^{d-1})$ in other words the d-dimensional density $I$ is projected into an infinite set of one-dimensional densities, $\mathcal{R}I(\cdot,\theta)$. The figure below visualizes the Radon transform for a 2D Gaussian density.

![$I_i$s and their corresponding CDT](Figures/figure1.png)

Using the Slice Fourier theorem it can be easily shown that the Radon transform is invertible. The inverse of the Radon transform denoted by $\mathcal{R}^{-1}$ is defined as:
$$
I(x)=\mathcal{R}^{-1}(\mathcal{R}I(t,\theta))= \int_{\mathbb{S}^{d-1}}\int_{\mathbb{R}} (\mathcal{R}I(t,\theta)*h(t))\delta(t-x\cdot\theta)dtd\theta
$$
where $h(.)$ is a one-dimensional high-pass filter with corresponding Fourier transform $\mathcal{F}h(\omega)\approx c|\omega|^{d-1}$ (it appears due to the change of coordinates from spherical to Cartesian in the Fourier Slice theorem) and `$*$' denotes convolution. The above definition of the inverse Radon transform is also known as the filtered back-projection method, which is extensively used in image reconstruction in the biomedical imaging community. Intuitively each one-dimensional projection/slice, $\mathcal{R}I(\cdot,\theta)$, is first filtered via a high-pass filter and then smeared back into $\mathbb{R}^{d}$ along the integration hyperplanes, $x\cdot\theta=t$, to approximate $I$. The summation of all smeared approximations then reconstruct $I$. Finally, we note that Radon transform is
a linear transformation.

### Radon-CDT

The idea behind Radon-CDT is to first slice a d-dimensional probability density $I$ into a set of one-dimensional distributions and then apply CDT to the one dimensional distributions. Formally, given a template distribution $I_0$ the Radon-CDT is defined as:

$$
\hat{I}(t,\theta)=\text{CDT}_{\mathcal{R}I_0}(\mathcal{R}I(\cdot,\theta))= (f(t,\theta)-t)\sqrt{\mathcal{R}I_0(t,\theta)}
$$

where $f(t,\theta)$ satisfies:

$$
\int_{-\infty}^{f(t,\theta)}\mathcal{R}I(\tau,\theta)d\tau=\int_{-\infty}^{t}\mathcal{R}I_0(\tau,\theta)d\tau,~ \forall \theta\in \mathbb{S}^{d-1}
$$

and the inverse transform is defined as:

$$
I(x)=\mathcal{R}^{-1}\left(\frac{\partial f(t,\theta)}{\partial t}\mathcal{R}I_0(f(t,\theta),\theta)\right)
$$

where $f(t,\theta)$ can be calculated from $\hat{I}$ and $I_0$ as  $f(t,\theta)=\frac{\hat{I}(t,\theta)}{\sqrt{\mathcal{R}I_0(t,\theta)}}+t$.

## Radon-CDT Demo

Throughout the experiments in this tutorial we assume that $\mathcal{R}I_0(\cdot,\theta)$ is a uniform one-dimensional distribution. Lets start by showing the nonlinear nature of Radon-CDT.

### Nonlinearity

Let $I_1$ and $I_2$ be the following two images:

![](Figures/I0.bmp) ![](Figures/I1.bmp)

where the images are normalized (sum to one) to be considered as two-dimensional probability distributions. Let $\hat{I}_i$ denote the corresponding Radon-CDTs. We use the following code to calculate the Radon transform and the Radon-CDT of these images (i.e. distributions).

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

![$I_i$s and their corresponding Radon-CDT](Figures/figure2.png)

Now to demonstrate the nonlinear nature of Radon-CDT, we choose the simplest linear operator, which is averaging the two distributions. We average the signals in the signal space, $I_3=0.5(I_1+I_2)$, in the Radon space $I_4=\mathcal{R}^{-1} (0.5(\mathcal{R}I_1+\mathcal{R}I_2))$, and in the Radon-CDT space, $I_5=\text{iRCDT}(0.5(\hat{I}_1+\hat{I}_2))$, and compare the results below.

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

Here we run a toy example to demonstrate this characteristic. We start by defining three classes of two-dimensional images, where Class $k$, for $k\in\{1,2,3\}$, consists of translated versions of a $k$-modal Gaussian distribution. Here we generate these signal classes and their corresponding Radon-CDTs.

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
