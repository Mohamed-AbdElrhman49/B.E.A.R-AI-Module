
# 🧠 B.E.A.R. – Bone Estimation and Age Reporting

B.E.A.R. is an AI-powered diagnostic assistant designed to help medical professionals **predict bone age** and detect **bone growth disorders** from hand X-ray images — in **under 30 seconds**.

Our system integrates **computer vision**, **deep learning**, and **natural language processing (NLP)** into a complete mobile application, providing **instant analysis**, **accurate age prediction**, and **auto-generated medical reports**.

---

## 🚀 Project Goals

- Early detection of bone growth disorders in children.
- Fast, accurate bone age estimation using X-rays.
- Providing doctors with a fully automated reporting system.
- Making diagnostics more accessible and reliable.

---

## 📊 Dataset

- **RSNA Bone Age Dataset** (~14,236 X-rays)
- **RHPE Dataset** (~14k images)

Images are hand X-rays of children with labeled ages and genders.

---

## 🧠 Technologies Used

### 🖼️ Image Processing
- **CLAHE**, **Otsu Thresholding**, **Canny Edges**
- **Median & Gaussian Blur** for denoising

### 🤖 Object Detection
- **YOLOv8** for hand localization in the X-ray image

### 📈 Age Estimation Models
- **CNNs**: ResNet18, ResNet50, DenseNet121
- **Transformers**: Vision Transformer (ViT)
- **Best Result**: **Xception** + YOLO Cropping → MAE = **6.34 months**

### 🧾 Report Generator
- **Meta’s LLaMA 7B** fine-tuned on medical text
- Generates structured, professional reports from predictions

### 📱 Mobile App
- Built with **Flutter**
- Backend: **Firebase** (auth, database, storage)
- Real-time predictions via **REST API**


### 1. Clone the repo
```bash
git clone https://github.com/your-username/B.E.A.R..git](https://github.com/Mohamed-AbdElrhman49/B.E.A.R-AI-Module.git
cd B.E.A.R.
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Start Backend Server
```bash
python app.py
```

### 4. (Optional) Run YOLOv8 Hand Detection
```bash
# Assuming using Ultralytics YOLOv8
yolo task=detect mode=predict model=yolov8n.pt source=your_xray_folder
```

### 5. Run Inference
Model will take the cropped hand image and output bone age prediction + report.

---

## 📈 Results

| Model | MAE (months) | RMSE |
|-------|--------------|------|
| ResNet50 | 9.80 | 12.23 |
| DenseNet121 | 8.63 | 10.34 |
| **Xception + YOLO** | **6.34** | **7.45** |

---

## 👨‍⚕️ Team

- Zeyad Omar  
- Sondos Amr  
- Mohammed Abodaif  
- Kareem Abdeen  
- Aya Nady  
- Mostafa Awad

🎓 Under the supervision of **Dr. [Fatma Mazen]**

---

## 📬 Contact

If you have any questions or suggestions, feel free to open an issue or reach out!

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
