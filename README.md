# 🩺 Pneumonia Detection from Chest X-rays  
**Deep Learning for Early Diagnosis**

---

## 📌 Overview

Pneumonia remains a leading cause of morbidity worldwide, and timely diagnosis is critical. This project presents an AI-driven solution that analyzes chest X-ray images to detect signs of pneumonia. By leveraging transfer learning with a pre-trained VGG16 model, the system classifies images as either **Normal** or **Pneumonia**, offering a scalable tool to assist radiologists and healthcare professionals.

✅ Built for medical imaging workflows  
✅ Streamlined training with VGG16 architecture  
✅ Achieves 82% training accuracy 

---

## 📁 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup & Installation](#setup--installation)
- [Training & Tuning](#training--tuning)
- [Results](#results)
- [Demo Video](#demo-video)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

---

## 🧬 Dataset

The model is trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset by Paul Mooney, hosted on Kaggle. It contains 5,863 labeled X-ray images divided into two categories:

| Class      | Image Count |
|------------|-------------|
| Normal     | 1,583       |
| Pneumonia  | 4,273       |

All images are resized to 224×224 pixels for compatibility with the VGG16 input layer.

---

## 🧠 Model Architecture

This project uses transfer learning to fine-tune a VGG16 model for binary classification:

| Component       | Description                                  |
|----------------|----------------------------------------------|
| Base Model     | VGG16 (pre-trained on ImageNet)              |
| Input Size     | 224×224 pixels                               |
| Output Layer   | Dense layer with 2 units + softmax           |
| Optimizer      | Adam                                          |
| Loss Function  | Categorical Crossentropy                     |
| Epochs         | 5 (configurable)                             |

The base model’s convolutional layers are frozen to retain learned features, while the top layers are retrained on the pneumonia dataset.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Im-Mohammed/PneumoniaDetectionModel.git
cd PneumoniaDetectionModel
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the dataset from Kaggle and extract it to:

```
./data/chest_xray/
```

---

## 🔧 Training & Tuning

To train the model:

```bash
python train.py
```

### Hyperparameter Tuning

You can customize the following parameters:

- **Batch Size**: Default is 32  
- **Learning Rate**: Example:  
  ```python
  optimizer = Adam(learning_rate=0.0001)
  ```
- **Epochs**: Modify in `model.fit()` as needed

---

## 📈 Results

- ✅ Training Accuracy: 82%  

The model demonstrates strong baseline performance and can be further improved with data augmentation, regularization, or deeper fine-tuning.

---

## 🎥 Demo Video

Watch the Pneumonia Detection model in action:

[▶️ Click to Watch Demo](https://github-production-user-asset-6210df.s3.amazonaws.com/128249314/361316701-409efd31-e61a-4b6e-abd4-6e68d23e8e05.mp4)

> _Note: This video is hosted on GitHub's asset server and may expire. For long-term access, consider uploading to YouTube or Vimeo._

---

## 🩻 Usage

After training, run predictions on new chest X-ray images:

```bash
python predict.py --image path/to/image.jpg
```

The script will output one of the following classifications:

- **Normal**
- **Pneumonia**

---

## 🤝 Contributing

Contributions are welcome and appreciated. To contribute:

```bash
# Fork the repository
git checkout -b feature-branch
git commit -m "Add new feature"
git push origin feature-branch
```

Then open a pull request with a clear description of your changes.

---

## 📬 Contact

For questions, feedback, or collaboration inquiries:  
📮 [GitHub Issues](https://github.com/Im-Mohammed/PneumoniaDetectionModel/issues)

---

## 📄 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute with attribution.
