from PIL import Image
import glob


training_filenames = glob.glob("../results/predictions_training/*.png")
test_filenames = glob.glob("../results/predictions_test/*.png")

def upsample_training(filenames, correct_size=(400, 400)):
    """Upsample training images."""
    print("Upsampling training images to %dx%d" % correct_size)
    resize_(filenames, correct_size)

def upsample_test(filenames, correct_size=(608, 608)):
    """Upsample test images"""
    print("Upsampling test images to %dx%d" % correct_size)
    resize_(filenames, correct_size)

def resize_(filenames, correct_size):
    """Resie given images to the correct size."""
    for filename in filenames:
        print("Resizing " + filename)
        im = Image.open(filename)
        im = im.resize(correct_size, Image.ANTIALIAS)
        im.save(filename, "PNG")