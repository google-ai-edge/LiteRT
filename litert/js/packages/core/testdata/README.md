# Test Data for LiteRT Web

The example models in this directory come from these sources:

* `multi_signature_model.tflite`: [This colab](https://colab.corp.google.com/drive/1qbbCn6erffX6MvmJkrKIbAPNHNxPnkLs)
* `add_c1_c7.tflite`: tf.add({shape=\[1\], dtype=tf.float32}, {shape=\[7\], dtype=tf.float32})
* `add_10x10.tflite`: tf.add({shape=\[10, 10\], dtype=tf.float32}, {shape=\[10, 10\], dtype=tf.float32})
* `add_1d_2d_3d_4d.tflite`: tf.add({shape=\[4\], dtype=tf.float32}, {shape=\[3, 4\], dtype=tf.float32}, {shape=\[2, 3, 4\], dtype=tf.float32}, {shape=\[1, 2, 3, 4\], dtype=tf.float32})
* `delegate_compatibility_test.tflite`: [This colab](https://colab.research.google.com/drive/1SiF9brI_qsdoMUh6p_N6WQYUH_Y1Ef9X?usp=sharing)
* `mixed_input_model.tflite`: [This colab](https://colab.sandbox.google.com/drive/1j9ZIFhjNyAVgs38C2puhX1huKGWGBk75#scrollTo=88f10dbd)
