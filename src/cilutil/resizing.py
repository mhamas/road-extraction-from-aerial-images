from PIL import Image
import glob


def upsample_training(filenames, correct_size=(400, 400)):
    """Upsample training images."""
    print("Upsampling training images to %dx%d" % correct_size)
    resize_(filenames, correct_size)


def upsample_test(filenames, correct_size=(608, 608)):
    """Upsample test images."""
    print("Upsampling test images to %dx%d" % correct_size)
    resize_(filenames, correct_size)


def downsample_training(filenames, correct_size=(200, 200)):
    """Downsample training images."""
    print("Upsampling training images to %dx%d" % correct_size)
    resize_(filenames, correct_size)


def downsample_test(filenames, correct_size=(304, 304)):
    """Upsample test images."""
    print("Upsampling test images to %dx%d" % correct_size)
    resize_(filenames, correct_size)


def resize_(filenames, correct_size):
    """Resize given images to the correct size."""
    for filename in filenames:
        print("Resizing " + filename)
        im = Image.open(filename)
        im = im.resize(correct_size, Image.ANTIALIAS)
        im.save(filename, "PNG")


if __name__ == "__main__":
    # Downsample training images
    files1 = glob.glob("../data/test_set/downsampled/*.png")
    files2 = glob.glob("../data/training/groundtruth/downsampled/*.png")
    files3 = glob.glob("../data/training/images/downsampled/*.png")

    downsample_test(files1)
    downsample_training(files2 + files3)
