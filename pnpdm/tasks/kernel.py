import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from numpy.random import uniform, triangular, beta
from math import pi
from pathlib import Path
from scipy.signal import convolve

# tiny error used for nummerical stability
eps = 0.1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def norm(lst: list) -> float:
    """L^2 norm of a list"""
    if not isinstance(lst, list):
        raise ValueError("Norm takes a list as its argument")

    if lst == []:
        return 0

    return (sum((i**2 for i in lst)))**0.5


def polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """Convert polar coordinates (r, θ) to complex numbers r·e^{iθ} (x + iy)."""
    return r * np.exp(1j * θ)


class Kernel(object):
    """[summary]
    Class representing a motion blur kernel of a given intensity.

    [description]
    Keyword Arguments:
            size {tuple} -- Size of the kernel in px times px
            (default: {(100, 100)})

            intensity {float} -- Float between 0 and 1.
            Intensity of the motion blur.

            :   0 means linear motion blur and 1 is a highly non linear
                and often convex motion blur path. (default: {0})

    Attribute:
    kernelMatrix -- Numpy matrix of the kernel of given intensity

    Properties:
    applyTo -- Applies kernel to image
               (pass as path, pillow image or np array)

    Raises:
        ValueError
    """

    def __init__(self, size: tuple = (100, 100), intensity: float=0):

        if not isinstance(size, tuple):
            raise ValueError("Size must be TUPLE of 2 positive integers")
        elif len(size) != 2 or type(size[0]) != type(size[1]) != int:
            raise ValueError("Size must be tuple of 2 positive INTEGERS")
        elif size[0] < 0 or size[1] < 0:
            raise ValueError("Size must be tuple of 2 POSITIVE integers")

        if type(intensity) not in [int, float, np.float32, np.float64]:
            raise ValueError("Intensity must be a number between 0 and 1")
        elif intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")

        self.SIZE = size
        self.INTENSITY = intensity
        self.SIZEx2 = tuple([2 * i for i in size])
        self.x, self.y = self.SIZEx2

        self.DIAGONAL = (self.x**2 + self.y**2)**0.5

        self.kernel_is_generated = False

    def _createPath(self):
        """Create a motion-blur path for the given intensity.
        """

        def getSteps():
            """[summary]
            Here we calculate the length of the steps taken by
            the motion blur
            [description]
            We want a higher intensity lead to a longer total motion
            blur path and more different steps along the way.

            Hence we sample

            MAX_PATH_LEN =[U(0,1) + U(0, intensity^2)] * diagonal * 0.75

            and each step: beta(1, 30) * (1 - self.INTENSITY + eps) * diagonal)
            """

            # getting max length of blur motion
            self.MAX_PATH_LEN = 0.5 * self.DIAGONAL * \
                (uniform() + uniform(0, self.INTENSITY**2))

            # getting step
            steps = []

            while sum(steps) < self.MAX_PATH_LEN:

                # sample next step
                step = beta(1, 30) * (1 - self.INTENSITY + eps) * self.DIAGONAL
                if step < self.MAX_PATH_LEN:
                    steps.append(step)

            # note the steps and the total number of steps
            self.NUM_STEPS = len(steps)
            self.STEPS = np.asarray(steps)

        def getAngles():
            """[summary]
            Gets an angle for each step
            [description]
            The maximal angle should be larger the more
            intense the motion is. So we sample it from a
            U(0, intensity * pi)

            We sample "jitter" from a beta(2,20) which is the probability
            that the next angle has a different sign than the previous one.
            """


            self.MAX_ANGLE = uniform(0, self.INTENSITY * pi)

            self.JITTER = beta(2, 20)
            angles = [uniform(low=-self.MAX_ANGLE, high=self.MAX_ANGLE)]

            while len(angles) < self.NUM_STEPS:

                angle = triangular(0, self.INTENSITY *
                                   self.MAX_ANGLE, self.MAX_ANGLE + eps)

                if uniform() < self.JITTER:
                    angle *= - np.sign(angles[-1])
                else:
                    angle *= np.sign(angles[-1])

                angles.append(angle)

            self.ANGLES = np.asarray(angles)

        getSteps()
        getAngles()
        complex_increments = polar2z(self.STEPS, self.ANGLES)

        self.path_complex = np.cumsum(complex_increments)

        self.com_complex = sum(self.path_complex) / self.NUM_STEPS

        center_of_kernel = (self.x + 1j * self.y) / 2
        self.path_complex -= self.com_complex

        self.path_complex *= np.exp(1j * uniform(0, pi))

        self.path_complex += center_of_kernel

        self.path = [(i.real, i.imag) for i in self.path_complex]

    def _createKernel(self, save_to: Path=None, show: bool=False):
        """Create the motion-blur kernel image (if not already generated)."""

        if self.kernel_is_generated:
            return None

        self._createPath()

        self.kernel_image = Image.new("RGB", self.SIZEx2)

        # ImageDraw instance that is linked to the kernel image that
        self.painter = ImageDraw.Draw(self.kernel_image)

        # draw the path
        self.painter.line(xy=self.path, width=int(self.DIAGONAL / 150))

        # applying gaussian blur for realism   
        self.kernel_image = self.kernel_image.filter(
            ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01)))

        # Resize to actual size
        self.kernel_image = self.kernel_image.resize(
            self.SIZE, resample=Image.LANCZOS)

        # convert to gray scale
        self.kernel_image = self.kernel_image.convert("L")

        # flag that we have generated a kernel
        self.kernel_is_generated = True


    def displayKernel(self, save_to: Path=None, show: bool=True):
        """[summary]
        Finds a kernel (psf) of given intensity.
        [description]
        Saves the kernel to save_to if needed or shows it
        is show true

        Keyword Arguments:
            save_to {Path} -- Image file to save the kernel to. {None}
            show {bool} -- shows kernel if true
        """

        self._createKernel()

        if save_to is not None:
            save_to_file = Path(save_to)
            self.kernel_image.save(save_to_file)
        else:
            self.kernel_image.show()

    @property
    def kernelMatrix(self) -> np.ndarray:
        """[summary]
        Kernel matrix of motion blur of given intensity.
        [description]
        Once generated, it stays the same.
        Returns:
            numpy ndarray
        """

        self._createKernel()
        kernel = np.asarray(self.kernel_image, dtype=np.float32)
        kernel /= np.sum(kernel)

        return kernel

    @kernelMatrix.setter
    def kernelMatrix(self, *kargs):
        raise NotImplementedError("Can't manually set kernel matrix yet")

    def applyTo(self, image, keep_image_dim: bool = False) -> Image:
        """[summary]
        Applies kernel to one of the following:
            1. Path to image file
            2. Pillow image object
            3. (H,W,3)-shaped numpy array
        [description]
        Arguments:
            image {[str, Path, Image, np.ndarray]}
            keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

        Returns:
            Image -- [description]
        """
        # calculate kernel if haven't already
        self._createKernel()

        def applyToPIL(image: Image, keep_image_dim: bool = False) -> Image:
            """[summary]
                Applies the kernel to an PIL.Image instance
            [description]
                converts to RGB and applies the kernel to each
                band before recombining them.
            Arguments:
                image {Image} -- Image to convolve
                keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

            Returns:
                Image -- blurred image
            """

            image = image.convert(mode="RGB")

            conv_mode = "valid"
            if keep_image_dim:
                conv_mode = "same"

            result_bands = ()

            for band in image.split():

                # convolve each band individually with kernel
                result_band = convolve(
                    band, self.kernelMatrix, mode=conv_mode).astype("uint8")

                result_bands += result_band,

            result = np.dstack(result_bands)

            return Image.fromarray(result)

        if isinstance(image, str) or isinstance(image, Path):

            image_path = Path(image)
            image = Image.open(image_path)

            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, Image.Image):

            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

            return applyToPIL(image, keep_image_dim)

        else:

            raise ValueError("Cannot apply kernel to this type.")


if __name__ == '__main__':
    image = Image.open("./images/moon.png")
    image.show()
    k = Kernel()

    k.applyTo(image, keep_image_dim=True).show()