# 🩺 Pneumonia Detection from Chest X-rays  
**AI-Powered Diagnostic Support Using Transfer Learning**

Pneumonia is a serious respiratory condition that demands timely and accurate diagnosis. This project introduces an intelligent system that analyzes chest X-ray images to detect signs of pneumonia using deep learning. By fine-tuning a pre-trained VGG16 model, the solution offers fast, reliable classification of X-rays as either **Normal** or **Pneumonia**, supporting healthcare professionals in early intervention.

Built with clarity, precision, and real-world utility in mind, this tool demonstrates how AI can enhance diagnostic workflows and reduce the burden on clinical staff.

---

## 🧬 Dataset

The model is trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset, which contains 5,863 labeled images:

- **Normal**: 1,583 images  
- **Pneumonia**: 4,273 images

All images are resized to 224×224 pixels to match the input requirements of the VGG16 architecture.

---

## 🧠 Model Highlights

This project uses transfer learning to adapt the VGG16 convolutional neural network for binary classification:

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Input Size**: 224×224 pixels
- **Output Layer**: Dense layer with 2 units + softmax
- **Epochs**: 5 (configurable)

The convolutional layers of VGG16 are frozen to preserve learned features, while the top layers are retrained on the pneumonia dataset.

---

## ⚙️ Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Im-Mohammed/PneumoniaDetectionModel.git
cd PneumoniaDetectionModel
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it in:

```
./data/chest_xray/
```

---

## 🔧 Training the Model

To begin training:

```bash
python train.py
```

You can adjust key hyperparameters such as batch size, learning rate, and number of epochs:

```python
optimizer = Adam(learning_rate=0.0001)
```

Modify the `model.fit()` call to experiment with different training durations.

---

## 📈 Performance

The model achieves a training accuracy of **82%**, offering a strong baseline for pneumonia detection. With further tuning and data augmentation, this performance can be improved for deployment-ready use cases.

---

## 🎥 Demo Video

Experience the model in action:

[▶️ Click to Watch Demo](https://github-production-user-asset-6210df.s3.amazonaws.com/128249314/361316701-409efd31-e61a-4b6e-abd4-6e68d23e8e05.mp4)

> _Note: This video is hosted on GitHub's asset server and may expire. For long-term access, consider uploading to YouTube or Vimeo._

---

## 🩻 Predicting New Images

Once trained, you can run predictions on new chest X-ray images:

```bash
python predict.py --image path/to/image.jpg
```

The output will classify the image as either:

- **Normal**
- **Pneumonia**

---

## 🤝 Collaboration

This project welcomes contributions from researchers, developers, and healthcare innovators. If you’d like to improve the model, add new features, or adapt it for broader use cases, feel free to fork the repo and submit a pull request:

```bash
git checkout -b feature-branch
git commit -m "Add new feature"
git push origin feature-branch
```

---

## 📬 Contact

For questions, feedback, or collaboration opportunities:  
📮 [Open an Issue](https://github.com/Im-Mohammed/PneumoniaDetectionModel/issues)

---

## 📄 License

This project is licensed under the **MIT License**.  
You’re free to use, modify, and distribute it with attribution.
