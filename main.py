
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("here")

    # begin
    from keras.applications.vgg16 import VGG16

    model = VGG16()
    model.summary()

    print(len(model.layers))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
