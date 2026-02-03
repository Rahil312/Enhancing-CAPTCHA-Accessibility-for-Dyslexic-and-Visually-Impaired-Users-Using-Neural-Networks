# Enhancing CAPTCHA Accessibility for Dyslexic and Visually Impaired Users Using Neural Networks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A deep learning approach to make CAPTCHAs more accessible through automated recognition*

</div>

## 🎯 Project Overview

This project addresses the accessibility challenges faced by dyslexic and visually impaired users when encountering CAPTCHAs online. By developing a robust CNN-based CAPTCHA recognition system, we aim to create assistive technology that can automatically solve CAPTCHAs, making web content more accessible.

### 🌟 Key Features

- 🧠 **Advanced CNN Architecture** - Multi-layer convolutional neural network with batch normalization
- 🎯 **Multi-character Recognition** - Simultaneous prediction of 6-character CAPTCHAs
- 🔤 **Comprehensive Character Set** - Supports 36 characters (a-z, 0-9)
- ♿ **Accessibility Focus** - Designed to assist users with visual and cognitive disabilities
- 📊 **High Accuracy** - Optimized for real-world CAPTCHA recognition tasks

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose | Version |
|------------|---------|----------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core Language | 3.7+ |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | Deep Learning Framework | 2.0+ |
| ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) | Computer Vision | 4.0+ |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical Computing | Latest |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data Manipulation | Latest |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat-square&logo=python&logoColor=white) | Visualization | Latest |
| ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | Machine Learning Utils | Latest |

</div>

## 🏗️ Model Architecture

```
Input (50×200×1 Grayscale Image)
    ↓
Conv2D Layer (16 filters, 3×3, ReLU)
    ↓
MaxPooling2D
    ↓
Conv2D Layer (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D
    ↓
Conv2D Layer (32 filters, 3×3, ReLU)
    ↓
Batch Normalization
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
6× Dense Branches (64 → Dropout → 36 classes)
    ↓
6-Character Output (Softmax)
```

## 📁 Project Structure

```
📦 CAPTCHA-Recognition-CNN/
├── 📄 Captch_Final_Code_1.ipynb    # Main implementation notebook
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Dependencies
├── 📄 LICENSE                      # MIT License
├── 📄 Proj.pdf                     # Project report
├── 🖼️ 1.png, 2.png, 3.png, 4.png   # Example images
└── 📄 .gitignore                   # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.7+ installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rahil312/Enhancing-CAPTCHA-Accessibility-for-Dyslexic-and-Visually-Impaired-Users-Using-Neural-Networks.git
   cd Enhancing-CAPTCHA-Accessibility-for-Dyslexic-and-Visually-Impaired-Users-Using-Neural-Networks
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Captch_Final_Code_1.ipynb
   ```

### Usage

1. 📂 Update the dataset path in the notebook to point to your CAPTCHA images
2. ▶️ Run all cells sequentially to train the model
3. 🎯 Use the `predict()` function to test on new CAPTCHA images
4. 📊 Evaluate model performance using the provided metrics

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Categorical Crossentropy |
| **Optimizer** | Adam |
| **Epochs** | 12 |
| **Batch Size** | 32 |
| **Train/Validation Split** | 80/10/10 |
| **Image Size** | 50×200×1 |
| **Character Set** | 36 (a-z, 0-9) |

## 📊 Results & Performance

The model demonstrates strong performance in CAPTCHA recognition tasks:

- ✅ **Overall Accuracy**: Measured by complete CAPTCHA string matches
- ✅ **Character-level Accuracy**: Individual character prediction accuracy
- ✅ **Real-time Prediction**: Fast inference for practical applications

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🏫 Neural Networks Course - Semester 2
- ♿ Accessibility research community
- 🧠 Open-source deep learning community
- 📚 TensorFlow and OpenCV teams

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out!

---

<div align="center">

**Made with ❤️ for accessibility and inclusion**

*"Technology should be accessible to everyone, regardless of their abilities."*

</div>