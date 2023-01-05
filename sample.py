import tensorflow as tf

import matplotlib.pyplot as plt

def image_enchant_features(image):

	# image = tf.image.resize(image, [128,128])[:,:,:3]
	image = tf.expand_dims(image, axis=0)

	model_normalize = tf.keras.models.Sequential([	
			tf.keras.layers.InputLayer(input_shape=(image.shape[1], image.shape[2])),
			tf.keras.layers.Normalization(mean=3., variance=2.),
			tf.keras.layers.Normalization(mean=4., variance=6.),
		])

	red, green, blue = image[:,:,:,0], image[:,:,:,1], image[:,:,:,2]

	predictions_red = tf.expand_dims( tf.squeeze( model_normalize.predict(red) ), axis=2 )
	predictions_green = tf.expand_dims( tf.squeeze( model_normalize.predict(green) ), axis=2 )
	predictions_blue = tf.expand_dims( tf.squeeze( model_normalize.predict(blue) ), axis=2 )
	
	# image = tf.experimental.numpy.vstack([predictions_red, predictions_green, predictions_blue])
	image = tf.concat([ predictions_red, predictions_green, predictions_blue ], axis=2)
	image = tf.keras.utils.array_to_img(image)

	return image
	
image_1 = plt.imread('F:\\temp\\20220117\\GO48R.png')
image_1 = plt.imread('F:\\Pictures\\Friends\\180102_10150135602181077_4952920_n.jpg')
image_2 = image_enchant_features( image_1 )

plt.figure(figsize=(1,2))
plt.subplot(1,2,2)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(image_2)
plt.xlabel('predictions')

plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(image_1)
plt.xlabel('image')

plt.show()
plt.close()
