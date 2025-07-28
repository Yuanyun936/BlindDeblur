'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''
import torch
from abc import ABC, abstractmethod


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

    def likelihood_gradient(self, data, y, sigma, **kwargs):
        return self.transpose(self.forward(data, **kwargs) - y).reshape(*data.shape) / sigma**2


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='no_noise')
class NoNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    @property
    def display_name(self):
        return f'no_noise_sigma={self.sigma}'
    
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma, input_snr=None):
        self.sigma = sigma
        self.input_snr = input_snr
    
    @property
    def display_name(self):
        if self.input_snr is None:
            return f'isigma=sigma={self.sigma}'
        else:
            return f'isnr={self.input_snr}_sigma={self.sigma}'
    
    def forward(self, data):
        if self.input_snr is None:
            return data + torch.randn_like(data, device=data.device) * self.sigma
        else:
            noise = torch.randn_like(data)
            noise_norm = torch.norm(data) * 10 ** (-self.input_snr / 20)
            scale = noise_norm / torch.norm(noise)
            scaled_noise = noise * scale
            print(f'input snr mode of gaussian noise: {scale} in input sigma')
            return data + scaled_noise


from .motion_blur import MotionBlurCircular
# from .retinal_blur import RetinalBlurCircular
from .kernel import Kernel