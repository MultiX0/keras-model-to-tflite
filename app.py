import tensorflow as tf
import tensorflow_hub as hub

class USELayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(USELayer, self).__init__(**kwargs)
        self.use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", input_shape=[], dtype=tf.string)

    def call(self, inputs):
        return self.use_layer(inputs)
    
    # Create an input layer
input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_layer")

# Create a custom layer that wraps the hub.KerasLayer
use_layer = USELayer()(input_layer)


# Create a Keras Model
model = tf.keras.Model(inputs=input_layer, outputs=use_layer)

# Print model summary
model.summary()

# Save the model in the new Keras format (recommended)
model.save('model.keras')

# Load the model (specifying custom objects)
model = tf.keras.models.load_model('model.keras', custom_objects={'USELayer': USELayer}, compile=False)

# Convert the model to TensorFlow Lite format using the Flex converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True  # Enables the new converter that supports more complex operations
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Built-in TensorFlow Lite operations
    tf.lite.OpsSet.SELECT_TF_OPS  # Allow TensorFlow operations (Flex)
]
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with tf.io.gfile.GFile('model_flex.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format with Flex ops.")
