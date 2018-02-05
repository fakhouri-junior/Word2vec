import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

vau = "hello,world"
my_string_tensor = tf.convert_to_tensor(vau)
sparse_tensor = tf.string_split([my_string_tensor],',')
my_values = sparse_tensor.values
print(sparse_tensor)
print(my_values[0])
print(my_values[1])
