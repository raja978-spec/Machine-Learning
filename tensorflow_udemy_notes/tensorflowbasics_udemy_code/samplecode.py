import tensorflow as tf

tensor = tf.constant("Hello, tensorflow!")

with tf.compat.v1.Session() as sess:
    output = sess.run(tensor)
    print(output.decode("utf-8"))