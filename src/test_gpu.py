import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("\nDispositifs disponibles:")
print(tf.config.list_physical_devices())

# Test simple pour vérifier l'accélération matérielle
with tf.device('/device:GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    
print("\nCalcul matriciel effectué avec succès!")
print("Si vous voyez cette message sans erreur, l'accélération matérielle fonctionne.")