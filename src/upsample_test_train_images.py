from PIL import Image
import glob


training_filenames = glob.glob("../results/predictions_training/*.png")
test_filenames = glob.glob("../results/predictions_test/*.png")

train_correct_size = 400, 400
for filename in training_filenames:
    print("Resizing " + filename)
    im = Image.open(filename)
    im = im.resize(train_correct_size, Image.ANTIALIAS)
    im.save(filename, "PNG")

test_correct_size = 608, 608
for filename in test_filenames:
    print("Resizing " + filename)
    im = Image.open(filename)
    im = im.resize(test_correct_size, Image.ANTIALIAS)
    im.save(filename, "PNG")