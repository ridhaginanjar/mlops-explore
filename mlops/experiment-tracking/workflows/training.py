from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train(train_dir, test_dir, MODELS, OPTIMIZERS: str, LOSSES: str, METRICS: list, callbacks=None):
    train_datagen = ImageDataGenerator(
        rescale= 1/.255,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(
        rescale=1/.255,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        subset='training',
        class_mode='binary',
        target_size=[150,150],
        batch_size=32,
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        subset='validation',
        target_size=[150,150],
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=[150,150],
        batch_size=32,
        class_mode='binary',
        shuffle=False # because important to evaluate.
    )

    MODELS.compile(optimizer=OPTIMIZERS, loss=LOSSES, metrics=METRICS)

    EPOCH = 20

    history = MODELS.fit(
        train_gen,
        validation_data=val_gen,
        epochs= EPOCH,
        callbacks=[callbacks]
    )

    return MODELS, train_gen, test_gen