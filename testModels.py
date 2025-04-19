import tensorflow as tf
import cv2
from keras.utils import load_img, img_to_array
import numpy as np
import os
import matplotlib.pyplot as plt

IMAGE_SIZE = 224

train_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\train"
val_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\val"
test_dir = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\test"

CLASSES = os.listdir(train_dir)
num_classes = len(CLASSES)
# "C:\Users\sumhs\Documents\Projects\FYP\temp\BreakHis-ResNet50.keras"
best_model_file = r"C:\Users\sumhs\Documents\Projects\FYP\temp\BreakHis-VGG16.keras"
model = tf.keras.models.load_model(best_model_file)

def get_gradcam_heatmap(model, img_array, class_index, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, class_index]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = np.dot(conv_outputs, pooled_grads.numpy()) 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def apply_gradcam(image_path, heatmap, alpha=0.5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.show()

    return superimposed_img

def prepareImage(imagePath):
    image = load_img(imagePath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.0
    return imgResult

testImagePath = r"C:\Users\sumhs\Documents\Projects\BreastCancer\dataset_split2_200X\test\malignant\SOB_M_DC-14-3909-200-016.png"

img = cv2.imread(testImagePath)

imgForModel = prepareImage(testImagePath)

resultArray = model.predict(imgForModel, verbose=1)
print(resultArray)

answer = np.argmax(resultArray, axis=1)  # retrieve the higher probability item
print(answer)

heatmap = get_gradcam_heatmap(model, imgForModel, answer[0], layer_name="block5_conv3")

overlayed_image = apply_gradcam(testImagePath, heatmap)

index = answer[0]
className = CLASSES[index]

print(f"The predicted class is: {className}")

cv2.putText(img, className, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("img", img)
cv2.waitKey(0)