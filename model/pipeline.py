def train_model():
    # import library
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import tensorflow as tf
    from tensorflow.keras.callbacks import Callback, EarlyStopping
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
    import time, os

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.applications import MobileNetV2
    
    # Import path gambar
    train_dir = r"C:/Users/ACER/OneDrive/Documents/GitHub/sistem-absensi-berbasis-face-recognition/model/Face Images/Final Training Images"
    test_dir = r"C:/Users/ACER/OneDrive/Documents/GitHub/sistem-absensi-berbasis-face-recognition/model/Face Images/Final Testing Images"

    ### TRAIN MODEL ##
    # Praproses & augmentasi
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    #-----------------------
    # Bangun model
    # Definisikan callback kustom untuk menghentikan pelatihan ketika akurasi > 96%
    class CustomEarlyStopping(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy') >= 0.96:
                print(f"/nAkurasi validasi mencapai {logs.get('val_accuracy') * 100:.2f}%, menghentikan pelatihan!")
                self.model.stop_training = True
                
    # Load the MobileNetV2 model tanpa top layer
    mobilenet_base = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Freeze the base model
    mobilenet_base.trainable = False

    # Membangun model CNN
    Model = Sequential([
        mobilenet_base,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.3),  # Dropout layer untuk mengurangi overfitting
        Dense(train_generator.num_classes, activation='softmax')
    ])

    # Kompilasi model
    Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Menggunakan Early stopping untuk mengurangi waktu pelatihan
    call = CustomEarlyStopping()

    # Mengukur waktu yang dibutuhkan oleh model untuk melatih
    StartTime = time.time()

    # Melatih model
    history = Model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        epochs=20,
        callbacks=[call])

    Endtime = time.time()
    print('Total Training Time taken: ', round((Endtime - StartTime) / 60), 'Minutes')

    #######
    # Simpan model
    Model.save('C:/Users/ACER/OneDrive/Documents/GitHub/sistem-absensi-berbasis-face-recognition/model/Model_Face_Recognition_mobilenet.h5')
    print("Model retrained successfully.")
