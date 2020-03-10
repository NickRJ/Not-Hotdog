# Not Hotdog
Implemented Silicon Valley's [Not Hotdog](https://apps.apple.com/us/app/not-hotdog/id1212457521) classifier for fun. I used Tensorflow and Google Cloud's [Deep Learning VM](https://console.cloud.google.com/marketplace/config/click-to-deploy-images/tensorflow) for training and experimenting with different models.

### Dataset
Wanted to to do this without use of pretrained networks, so I manually gathered ~13,000 images from [ImageNet](http://www.image-net.org), [COCO](http://cocodataset.org), [Google](https://github.com/hardikvasa/google-images-download), and [Bing](https://github.com/sczhengyabin/Image-Downloader) image search results. There are 6,808 images of hotdogs and 6,577 images of other types of foods. I used 1500 images for the validation set, and 1000 for the test set.

Each image is resized and cropped to 64x64px, then augmented with random rotation, zoom, shift, and horizontal flips.

![dataset](https://github.com/NickRJ/Not-Hotdog/blob/master/dataset.png)

### Model
```python
model = Sequential([
    Conv2D(32, kernel_size=5, strides=2, padding='same', input_shape=(64, 64, 3)),
    BatchNormalization(),
    ReLU(),
    
    Conv2D(64, kernel_size=5, strides=2, padding='same'),
    BatchNormalization(),
    ReLU(),
    Dropout(0.3),
    
    Conv2D(64, kernel_size=5, strides=2, padding='same'),
    BatchNormalization(),
    ReLU(),
    Dropout(0.3),
    
    Flatten(),
    Dense(1024),
    BatchNormalization(),
    ReLU(),
    Dense(1, activation='sigmoid')
])
```

### Training
I used [Adam](https://arxiv.org/abs/1412.6980) with a learning rate of 1e-2. The model gets 85% accuracy on the test set without any use of pretrained networks. Finetuning a pretrained network would likely get 99% accuracy. Might do this in the future.

![accuracy](https://github.com/NickRJ/Not-Hotdog/blob/master/accuracy.png)
![loss](https://github.com/NickRJ/Not-Hotdog/blob/master/loss.png)
