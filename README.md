# blur_contrast_normalize_filter

For study blur and contrast normalize filter, simply apply filters on image with less efforts and readable codes in few lines.

## Normalization ##

Normalization is the distribution of values and makes it continuous when those values are existing but they are required to organize and aligned for senses that are because it is trends to be attractive when value is continuous than distributed.

Some applications are working with contrast environment background colors and significant see the deer forest example, it is better with single than multiply colors that is because colors itself blur the image but we select to see by our filter.

```
def image_enchant_features(image):

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
	
    image = tf.concat([ predictions_red, predictions_green, predictions_blue ], axis=2)
    image = tf.keras.utils.array_to_img(image)
    
    return image
```

## Files and Directory ##

| File Name | Description  |
--- | --- |
| sample.py | sample codes |
| Figure_6.png | image filetered |
| Figure_8.png | image filetered |
| Figure_9.png | image filetered |
| Figure_10.png | image filetered |
| 18.png | image filetered |
| README.md | readme file |

## Results ##

#### See the cartoons image contrast better effects on single colors ####

![Alt text](https://github.com/jkaewprateep/blur_contrast_normalize_filter/blob/main/Figure_6.png "Title")

#### Shadow and line on the duck stomach bench ####

![Alt text](https://github.com/jkaewprateep/blur_contrast_normalize_filter/blob/main/Figure_8.png "Title")

#### Background contrast ####

![Alt text](https://github.com/jkaewprateep/blur_contrast_normalize_filter/blob/main/Figure_9.png "Title")

#### Letter and circuits line ####

![Alt text](https://github.com/jkaewprateep/blur_contrast_normalize_filter/blob/main/Figure_10.png "Title")

#### Deer seasons forest ####

![Alt text](https://github.com/jkaewprateep/blur_contrast_normalize_filter/blob/main/18.png "Title")
