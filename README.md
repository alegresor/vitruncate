# vitruncate

Using variational inference to generate samples from truncated distributions.

## Stein Method


Based on **[1]**. 

Algorithm 1 of the above paper wants us to compute:$$\hat{\phi}^*(\boldsymbol{z})=\frac{1}{n}\sum_{i=1}^n\left[k(\boldsymbol{x}_i,\boldsymbol{z})\nabla_{\boldsymbol{x}_i}\log(p(\boldsymbol{x}_i))+\nabla_{\boldsymbol{x}_i} k(\boldsymbol{x}_i,\boldsymbol{z})\right]$$

Assume we are using the [RBF kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
$$k(\boldsymbol{x},\boldsymbol{z})=\exp(-\lVert \boldsymbol{z}-\boldsymbol{x} \rVert^2 /h)$$
with gradient 
$$\nabla_{\boldsymbol{x}}k(\boldsymbol{x},\boldsymbol{z})=\frac{2}{h}k(\boldsymbol{x},\boldsymbol{z})\sum_{i=1}^n (\boldsymbol{z}-\boldsymbol{x})$$

Also assume we want to generate from a truncated multi-variate Gaussian. Letting C be a normalizaiton constant, $(\boldsymbol{L},\boldsymbol{U})$ lower and upper bounds respectively, and $\chi$ the indicator function, then the truncated density is  
$$p(\boldsymbol{x}|\boldsymbol{\mu},\boldsymbol{\Sigma},\boldsymbol{L},\boldsymbol{U})=C(2\pi)^{-d/2}\det(\boldsymbol{\Sigma})^{-1/2}\exp(-(\boldsymbol{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})/2)\chi_{(\boldsymbol{L},\boldsymbol{U})}(\boldsymbol{x}),$$
the $\log$ density is
$$\log(p(\boldsymbol{x}|\boldsymbol{\mu},\boldsymbol{\Sigma},\boldsymbol{L},\boldsymbol{U}))=\left(log(C)-\frac{d}{2}\log(2\pi)-\frac{1}{2}\log(\det(\boldsymbol{\Sigma})) - \frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)\chi_{(\boldsymbol{L},\boldsymbol{U})},$$
and the gradient of the log density is 
$$\nabla_{\boldsymbol{x_j}} \log(p(\boldsymbol{x}|\boldsymbol{\mu},\boldsymbol{\Sigma},\boldsymbol{L},\boldsymbol{U})) = \left(\boldsymbol{\mu}^T\boldsymbol{\Sigma}^{-1} -\boldsymbol{x}^T\boldsymbol{\Sigma}^{-1}\right)\chi_{(\boldsymbol{L},\boldsymbol{U})}.$$

We can then update the particles such that
$$x \leftarrow x+\epsilon\hat{\phi}^*(\boldsymbol{x})$$

## References

**[1]** Liu, Qiang, and Dilin Wang. “Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm.” ArXiv:1608.04471 [Cs, Stat], Sept. 2019. http://arxiv.org/abs/1608.04471.