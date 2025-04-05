import gpytorch
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel=='matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

        elif kernel=='linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

        elif kernel=='periodic':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())

        elif kernel=='polynomial':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=5))

        elif kernel=='cosine':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())

        elif kernel=='piecewise_polynomial':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel())

        elif kernel=='polynomial_kernel_grad':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernelGrad(power=5))

        elif kernel=='spectral_mixture':
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4).cuda()
            self.covar_module.initialize_from_data(train_x, train_y)

        elif kernel=='spectral_delta_kernel':
            self.covar_module = gpytorch.kernels.SpectralDeltaKernel(num_dims=train_x.shape[-1], num_deltas=128).cuda()
            self.covar_module.initialize_from_data(train_x.cuda(), train_y.cuda())

        elif kernel=='arc':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.ArcKernel(base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)))

        elif kernel=='rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        elif kernel=='rbf_grad':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())

        elif kernel=='quadratic':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)