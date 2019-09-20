# Unet-Segmentation

La red neuronal unet nos sirve para generar nuevas imagenes a partir de una imagen dada como entrada. En este ejemplo veremos como entrenarla para que nos genera una mascara de segmentacion de los diferentes objetos que se encuentran en esta. Este ejemplo esta realizado con tensorflow y keras.

## Modelo

Este modelo funciona con varias capas convolucionales que usaremos para analizar la imagen dada siguiendo el siguiente esquema:

![perrete](https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/FCN1.png)

Para codificar la imagen utilizaremos la siguiente composicion de capas
```
def encoder_down(input, filters, dropout=False):
    x = Conv2D(filters, activation='relu', padding='same', kernel_initializer='he_normal', kernel_size=(3, 3))(input)
    x = Conv2D(filters,activation='relu', padding='same', kernel_initializer='he_normal', kernel_size=(3, 3))(x)
    p = MaxPooling2D(pool_size=(2, 2))(x)

    if dropout:
        p = Dropout(0.25)(p)

    return x, p
```
Para descodificar esta infrormacion utilizaremos la capa UpSampling de la siguiente forma.

```

def encoder_up(input, filter, skip):
    x = UpSampling2D(size=(2, 2))(input)
    concat = Concatenate()([x, skip])
    x = Conv2D(filter, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(concat)
    return Conv2D(filter, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
```

## Generacion Datos

El conjunto de datos hay que dividirlo en los directorios data/img y data/mask y en data/test/img y data/test/mask los usaremos para validar el negocio. 
Para generar las mascaras he utilizado la herramienta labelme

https://github.com/wkentaro/labelme

## Prediccion

Para hacer la prediccion cargamos los pesos guardados del entramiento con el metodo load_weights y con el metodo div_img_to_mask podemos separar la imagen en los diferentes elementos que contiene la mascara generada
