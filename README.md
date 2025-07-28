
# 🐾 Animal Image Classifier using VGG16

## 📖 Project Description

This project is an **AI-powered Animal Image Classifier** built using **Deep Learning** and **Transfer Learning with VGG16**. It predicts the type of animal from an image by leveraging a pre-trained VGG16 model fine-tuned for **15 animal classes**.

To make the project interactive, a **Streamlit web application** is provided for real-time predictions via **drag-and-drop image upload**.

This project is part of my internship with **Unified Mentor** as my **3rd project**. While not highly challenging, I built it to **sharpen my skills** and focus on practical implementation.

---

## 🚀 Features

✔ 15 Animal Classes: Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra
✔ Transfer Learning with **VGG16** for high accuracy
✔ **Data Augmentation** for better generalization
✔ **Streamlit Web UI** for drag-and-drop predictions
✔ **Model Evaluation** using Confusion Matrix and Accuracy Graphs

---

## 📂 Project Structure

```
├── dataset/
│   └── train/                # Folder with subfolders for each class
├── main.py                   # Script to train and save the model
├── app.py                    # Streamlit app for image classification
├── animal_classifier_vgg16.h5 # Trained model file
└── README.md
```

---

## 🛠️ Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
tensorflow
streamlit
numpy
pillow
matplotlib
seaborn
scikit-learn
```

---

## 📊 Model Training (main.py)

* Loads VGG16 as the base model (pre-trained on ImageNet)
* Adds custom Dense & Dropout layers
* Uses **Adam optimizer**, **categorical crossentropy**
* Trains for 10 epochs with **ImageDataGenerator**
* Saves model as `animal_classifier_vgg16.h5`

Run training:

```bash
python main.py
```

---

## 🌐 Web Application (app.py)

* Built using **Streamlit**
* Drag & drop an image or browse to upload
* Displays predicted animal name with confidence

Run the app:

```bash
streamlit run app.py
```

---

## ✅ How It Works

1. Train the model using `main.py` (or use pre-trained `.h5`)
2. Start the Streamlit app with `app.py`
3. Upload an image → Get prediction instantly!

---

## 📈 Evaluation

* **Confusion Matrix**
* **Accuracy over epochs**
* **Classification Report**

These are displayed during/after training.



## 🐍 Example Command

```bash
streamlit run app.py
```
