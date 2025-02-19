# %% [markdown]
# Importing all the dependencies

# %%
import numpy as np
import os
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.image as mpimg
import cv2
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


# %% [markdown]
# Getting the dataset

# %%
data =  r'C:\Users\user\Downloads\archive (1)\images'

train = os.path.join(data,'train')
test = os.path.join(data,'validation')

# %% [markdown]
# Using Data Augmentation

# %%
train_data_gen =  ImageDataGenerator(
    rescale = 1./255,
    rotation_range =20,
    zoom_range =0.2,
    horizontal_flip=True
)

test_data_gen = ImageDataGenerator(rescale=1./255)




# %%
train_generator = train_data_gen.flow_from_directory(
    train,
    target_size=(48,48),
    batch_size=128,
    class_mode = 'categorical',
    color_mode = 'grayscale'

)

test_generator = train_data_gen.flow_from_directory(
    test,
    target_size=(48,48),
    batch_size=128,
    class_mode = 'categorical',
    color_mode = 'grayscale'

)

# %% [markdown]
# Detecting classes of data

# %%
classes = list(train_generator.class_indices.keys())
classes

# %%
train_generator.class_indices

# %% [markdown]
# Plotting images from datas et

# %%
images , labels = next(train_generator)
plt.figure(figsize=(8,8))
for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(images[i].reshape(48,48),cmap='gray')
    plt.title(classes[np.argmax(labels[i])])
    plt.axis('off')
plt.show()


# %% [markdown]
# Now let's Build the CNN architecture

# %%
model = Sequential([
    Conv2D(64, (3,3),activation='relu',input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(254,activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(7,activation='softmax')

]
)

model.summary()

# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# %%
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
)

# %% [markdown]
# Plotting model's performance
# 

# %%
test_images , test_labels = next(test_generator)
predictions = model.predict(test_images)

y_true = np.argmax(test_labels,axis=1)
y_pred = np.argmax(predictions,axis=1)

print(classification_report(y_true,y_pred,target_names=classes))

cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.show()

# %% [markdown]
# Model is Prforming in decent way except for classes like disgust and fear because of unbalanced data

# %% [markdown]
# Now Detecting with unseen data

# %%
image_path = r"C:\Users\user\Downloads\archive (1)\images\validation\angry\9615.jpg"


def detection(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((48,48))
    image = np.asarray(image, dtype=np.float32) / 255.0 
    image = np.expand_dims(image, axis=0)  
    image = np.expand_dims(image, axis=-1)

    prediction = model.predict(image)

    predict_index = np.argmax(prediction)
    predict_class = classes[predict_index]

    confidence = round(prediction[0][predict_index] * 100,2)

    # Display the image with the predicted label and confidence
    plt.imshow(image[0, :, :, 0], cmap='gray')   
    plt.title(f'Predicted Emotion: {predict_class} ({confidence}%)')
    plt.axis('off')
    plt.show()

    return predict_class, confidence

predict_class, confidence_score = detection(image_path)
print(f"Predicted Class: {predict_class}, Confidence: {confidence_score}%")


# %%
model.save('Facial_expression_detection.h5')

# %%



