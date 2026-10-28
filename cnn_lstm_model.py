
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(
    sequence_length=16,
    image_height=96,
    image_width=96,
    num_classes=3,
    use_pretrained=True
):
    """
    AEGIS driver-vigilance classifier.

    Input:
        Video sequences shaped (batch, 16, 96, 96, 3)
        Pixel values normalized to [0, 1]

    Output:
        Class probabilities:
        0 = Alert
        1 = Low Vigilant
        2 = Drowsy
    """

    # ==================== INPUT ====================

    inputs = layers.Input(
        shape=(
            sequence_length,
            image_height,
            image_width,
            3
        ),
        name="video_frames"
    )

    # Convert [0, 1] pixel values to [-1, 1].
    x = layers.TimeDistributed(
        layers.Rescaling(scale=2.0, offset=-1.0),
        name="normalise_frames"
    )(inputs)

    # ==================== PRETRAINED CNN ====================

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_height, image_width, 3),
        include_top=False,
        weights="imagenet" if use_pretrained else None,
        pooling="avg"
    )

    # Initially freeze all CNN layers.
    backbone.trainable = False

    # Keep Batch Normalization layers frozen during fine-tuning.
    # Unfreeze only the final feature-extraction layers.
    if use_pretrained:
        backbone.trainable = True

        fine_tune_from = max(0, len(backbone.layers) - 30)

        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False

        for layer in backbone.layers[fine_tune_from:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    x = layers.TimeDistributed(
        backbone,
        name="frame_feature_extractor"
    )(x, training=False)

    # ==================== TEMPORAL LEARNING ====================

    x = layers.LSTM(
        64,
        return_sequences=True,
        dropout=0.3,
        name="temporal_lstm"
    )(x)

    # Learn the importance of each frame.
    scores = layers.Dense(
        1,
        activation="tanh",
        name="attention_scores"
    )(x)

    weights = layers.Softmax(
        axis=1,
        name="attention_weights"
    )(scores)

    weighted_features = layers.Multiply(
        name="weighted_temporal_features"
    )([x, weights])

    context = layers.Lambda(
        lambda values: tf.reduce_sum(values, axis=1),
        output_shape=(64,),
        name="temporal_attention"
    )(weighted_features)

    # ==================== CLASSIFICATION ====================

    x = layers.Dense(
        32,
        activation="relu",
        name="classifier_dense"
    )(context)

    x = layers.Dropout(
        0.4,
        name="classifier_dropout"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="vigilance_class"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="AEGIS_CNN_LSTM_Attention"
    )

    # Low learning rate for stable fine-tuning.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()

    trainable = sum(
        int(tf.size(variable).numpy())
        for variable in model.trainable_weights
    )

    total = sum(
        int(tf.size(variable).numpy())
        for variable in model.weights
    )

    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")


