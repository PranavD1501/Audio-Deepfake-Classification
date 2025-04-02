# **Audio-Deepfake-Classification**
**AI Speech Forgery Detection**

## **Overview**
This project focuses on detecting AI-generated speech using deep learning models optimized for real-time or near real-time detection. It analyzes real conversations and identifies synthetic speech with high accuracy.

---

## **Approaches Used for AI Speech Forgery Detection**

### **1. Wave2Vec2 Model (Fine-Tuned)**

#### **Key Technical Innovations:**
- Transformer-based self-supervised speech learning on raw audio data.
- Model can be fine-tuned for deepfake audio detection.

#### **Reported Performance Metrics:**
- **Equal Error Rate (EER):** 3-5%
- **Area Under Curve (AUC):** 95-98%

#### **Significance of Choosing This Model:**
- Wave2Vec2 supports **real-time inference**.
- The model is **robust** against deepfake audios.
- No need for additional feature extraction; **raw audio waveforms** are used directly.

#### **Limitations:**
- **High training cost** on large datasets.
- Requires **fine-tuning** for best results.

---

### **2. Speech Enhancement GAN for Deepfake Detection**

#### **Key Technical Innovations:**
- Uses **GANs (Generative Adversarial Networks)** to remove extra noise and expose deepfake audios.

#### **Reported Performance Metrics:**
- **EER:** 4-7%
- Performs well across **various environments**.

#### **Significance of Choosing This Model:**
- Effective in **real-world noisy conversations**.
- Can be **used alongside other models** to enhance detection.

#### **Limitations:**
- **High latency** due to complex GAN implementation.
- Requires **large datasets** to be efficient.

---

### **3. RawNet2 (CNN-Based Model)**

#### **Key Technical Innovations:**
- Uses **raw audio waveforms** directly with a CNN-based model.

#### **Reported Performance Metrics:**
- **EER:** ~3.5%
- Strong generalization across different datasets.

#### **Significance of Choosing This Model:**
- **Computationally efficient** compared to transformers.
- Works directly on **raw audio**, eliminating the need for feature engineering.

#### **Limitations:**
- More **complex to understand** compared to transformer models.
- **Struggles with low-quality audio** recordings.

---

## **Implementation Process**

### **Challenges Encountered & Solutions**

#### **1. Differentiating High-Quality AI-Generated Speech from Real Human Speech**
One of the most significant challenges was distinguishing high-quality AI-generated speech from real human speech, especially when AI-generated voices were trained on **large datasets** and enhanced with **prosody modeling** to introduce natural variations.

##### **Solutions:**
- Used **Wave2Vec2**, which processes raw audio waveforms rather than relying on handcrafted features like MFCCs.
- Fine-tuned the model on **recent AI-generated speech datasets**, including deepfake samples from **state-of-the-art TTS models**.
- Introduced **spectral analysis augmentation** to expose minor distortions that AI-generated voices often introduce.

#### **2. Imbalanced Dataset (More Real Speech Samples Than AI-Generated)**
Publicly available speech datasets contain a significantly larger proportion of **real human speech** compared to AI-generated samples, leading to model bias.

##### **Solutions:**
- Used **data augmentation** techniques such as adding noise, pitch shifts, and speed variations to increase diversity in AI-generated speech samples.
- Implemented **synthetic oversampling (SMOTE)** to balance the dataset.
- Applied **weighted loss functions** to penalize misclassification of AI-generated speech more heavily.

#### **3. Computational Complexity and Real-Time Processing Constraints**
Wave2Vec2 is a large **transformer-based model**, making it computationally expensive. For real-world applications, the system must operate in real time.

##### **Solutions:**
- Optimized inference using **quantization techniques** (e.g., FP16 precision and model pruning).
- Converted the trained model to **ONNX format** for efficient execution on low-power devices.
- Used **batch processing and dynamic padding** to speed up inference.

---

## **Assumptions Made**
- AI-generated speech contains **subtle artifacts** in frequency and prosody that can be learned by deep learning models.
- The dataset is **diverse enough** to represent real-world scenarios, including multiple languages, accents, and noise conditions.
- **Fine-tuning** a pretrained Wave2Vec2 model on AI-generated datasets will yield **higher accuracy** than training from scratch.

---

## **Model Selection and Analysis**
Wave2Vec2 was chosen due to its unique and efficient technical advantages:

1. Unlike traditional approaches that rely on feature extraction (e.g., spectrograms or MFCCs), **Wave2Vec2 processes raw waveforms**, making it more effective at detecting subtle distortions in AI-generated speech.
2. The model is **pretrained on massive datasets** of real speech, allowing it to learn deep audio representations. Fine-tuning on AI-generated speech enhances its ability to distinguish real vs. fake audio.
3. Unlike convolutional networks, **transformers capture long-range dependencies** in speech, improving detection accuracy in real-world conversations.
4. With **model optimizations**, the inference speed is ~50ms per sample, making it **suitable for live applications**.

---

## **Model Working (Technical Explanation)**
The **first stage** of the model is a **feature encoder**, which consists of **multiple 1D convolutional layers** that transform raw waveforms into a sequence of lower-dimensional speech representations. These convolutional layers act like a **filter bank**, capturing important **frequency and time-domain characteristics** of speech while discarding unnecessary noise.

Once the **feature encoder extracts basic audio representations**, they are passed into a **contextual transformer encoder**. This is the core innovation of Wave2Vec2, as it applies **self-attention mechanisms** to analyze long-range dependencies in speech. The transformer learns to model relationships between different speech segments, allowing it to capture variations in **intonation, pitch, and articulation**—features that distinguish real human speech from AI-generated speech.

For **AI-generated speech detection**, we fine-tune a pretrained Wave2Vec2 model on a dataset containing both **real human speech** and **AI-generated speech samples**. The final **classification layer** (a fully connected layer with softmax activation) determines whether the input speech is **human** or **AI-generated**.

Since Wave2Vec2 operates directly on **raw waveforms**, it detects **subtle inconsistencies** in AI-generated voices, such as **unnatural pitch modulation, synthetic harmonics, or timing irregularities**, which might be overlooked by traditional models.

---

## **Strengths and Weaknesses**

### **Strengths**
- ✅ Highly accurate in detecting AI-generated speech.
- ✅ Works across different AI models (e.g., **VALL-E, DeepfakeTTS, WaveNet**).
- ✅ Fast inference allows **real-time deployment**.

### **Weaknesses**
- ❌ May struggle with **highly sophisticated AI-generated speech**.
- ❌ Requires **high-end hardware** for best performance.
- ❌ **Computationally expensive** for edge devices.

---

## **Future Improvements**
- Enhancing dataset diversity with **new AI speech synthesis models**.
- Implementing **distillation techniques** to reduce model size for edge deployment.

---

## **Reflection Questions**

### **1. What were the most significant challenges in implementing this model?**
One of the biggest challenges was ensuring **correct preprocessing of raw audio data**. Since the model expects a specific sample rate (**16kHz**), all audio files had to be **resampled** correctly. Another challenge was handling **memory and computational efficiency**, as **Wave2Vec2 requires significant GPU resources** for training.

### **2. How might this approach perform in real-world conditions vs. research datasets?**
In research settings, the model performs well due to **clean and balanced datasets**. However, **background noise, audio compression, and speaker variations** in real-world scenarios can affect accuracy. Continuous retraining is necessary.

### **3. What additional data or resources would improve performance?**
More **diverse AI-generated speech datasets**, real-world audio from **podcasts, phone calls, and voice assistants**, and ensemble learning could further improve accuracy.

### **4. How would you deploy this model in production?**
Optimizing for **low-latency inference** using **quantization and pruning**, deploying via **APIs for streaming processing**, and **continuous retraining** will ensure robustness.
