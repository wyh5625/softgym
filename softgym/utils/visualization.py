import os
import cv2
import numpy as np
import imageio
import glob
from PIL import Image
from moviepy.editor import ImageSequenceClip, VideoClip
# from moviepy.video.io.VideoFileClip import VideoFileClip
# from moviepy.editor import VideoClip


# def make_grid(array, nrow=1, padding=0, pad_value=120, image_size=(256, 256)):
#     """ numpy version of the make_grid function in torch. Dimension of array: NHWC """
#     if len(array.shape) == 3:  # In case there is only one channel
#         array = np.expand_dims(array, 3)

#     N, H, W, C = array.shape
#     ncol = (N + nrow - 1) // nrow  # Calculate the number of columns

#     grid_height = nrow * image_size[0] + padding * (nrow - 1)
#     grid_width = ncol * image_size[1] + padding * (ncol - 1)

#     grid_img = np.full((grid_height, grid_width, C), pad_value, dtype=np.uint8)

#     idx = 0
#     for i in range(nrow):
#         for j in range(ncol):
#             if idx < N:
#                 y_start = i * (image_size[0] + padding)
#                 x_start = j * (image_size[1] + padding)
#                 img = array[idx]
#                 resized_img = cv2.resize(img, image_size)  # Resize the image to image_size
#                 grid_img[y_start:y_start + image_size[0], x_start:x_start + image_size[1]] = resized_img
#                 idx += 1

#     return grid_img


def make_grid(array, ncol=10, padding=0, pad_value=120, image_size=(256, 256)):
    """ numpy version of the make_grid function in torch. Dimension of array: NHWC """
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)

    N, H, W, C = array.shape
    
    # Calculate the number of rows based on the given number of columns
    nrow = N // ncol + 1 if N % ncol != 0 else N // ncol
    
    grid_height = nrow * image_size[0] + padding * (nrow - 1)
    grid_width = ncol * image_size[1] + padding * (ncol - 1)

    grid_img = np.full((grid_height, grid_width, C), pad_value, dtype=np.uint8)

    idx = 0
    for i in range(nrow):
        for j in range(ncol):
            if idx < N:
                y_start = i * (image_size[0] + padding)
                x_start = j * (image_size[1] + padding)
                img = array[idx]
                resized_img = cv2.resize(img, image_size)  # Resize the image to image_size
                grid_img[y_start:y_start + image_size[0], x_start:x_start + image_size[1]] = resized_img
                idx += 1

    return grid_img


if __name__ == '__main__':
    N = 12
    H = W = 50
    X = np.random.randint(0, 255, size=N * H * W* 3).reshape([N, H, W, 3])
    grid_img = make_grid(X, nrow=3, padding=5)
    cv2.imshow('name', grid_img / 255.)
    cv2.waitKey()


def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    gif_filename = fname + '.gif'
    mp4_filename = fname + '.mp4'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(gif_filename, fps=fps)
    clip.write_videofile(mp4_filename, fps=fps)
    return clip


def save_numpy_to_gif_matplotlib(array, filename, interval=50):
    from matplotlib import animation
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    def img_show(i):
        plt.imshow(array[i])
        print("showing image {}".format(i))
        return

    ani = animation.FuncAnimation(fig, img_show, len(array), interval=interval)


    ani.save('{}.mp4'.format(filename))

    import ffmpy
    ff = ffmpy.FFmpeg(
        inputs={"{}.mp4".format(filename): None},
        outputs={"{}.gif".format(filename): None})

    ff.run()
    # plt.show()

def save_frame_as_image(frame, filename, scale=1.0):
    """Saves a specific frame of an array of frames as an image using moviepy.
    
    Parameters
    ----------
    frame : array_like
        A specific frame from an array of frames
    filename : string
        The filename of the image to write to
    scale : float
        How much to rescale the image by (default: 1.0)
    """
    
    # ensure that the file has the desired image extension
    fname, ext = os.path.splitext(filename)
    if not ext:
        ext = ".png"
    filename = fname + ext
    
    # make the moviepy ImageClip
    clip = ImageSequenceClip([frame], fps=1).resize(scale)
    
    # Save the image
    clip.save_frame(filename)
