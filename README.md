# Audio-Deepfake-Classification
AI Speech Forgery Detection

Overview:
This project focuses on detecting AI-generated speech using deep learning models optimized for real-time or near real-time detection. It analyzes real conversations and identifies synthetic speech with high accuracy.

Approaches Used for AI Speech Forgery Detection:

1.Wave2Vec2 model (fine tuned):
  * Key-Technical Innovations:
      1.Transformer based self-supervised speech learning on data
      2.Model can be fine-tuned for Deepfake Audio detection.

  * Reported Performance Metrics:
      1.Equal Error Rate(EER):- 3-5%
      2.Area Under Curve(AUC):- 95-98%

  * Significance of Choosing this Model:
     1.Wave2Vec2 model has real time inference
     2.Model is Robust against Deepfake audios
     3.Conversion of Data not required raw audio waveforms used.

  * Limitations:
     1.High Training Cost on large datasets
     2.Requires Fine tuning for best results.

2.Speech Enhancement GAN for Deepfake Detection:
  * Key Technical Innovation:
     1.Uses GAN to remove extra noise and expose deepfake audios.

  * Reported Performance Metrics:
    1.Equal Error rate (EER):- 4-7%
    2.Works well in almost all environments.

  * Significance of Choosing this Model:
    1.Effective in real world conversations that contains noise.
    2.Can work alongside with other models

  * Limitations:
    1.High Latency due to complex GAN implementation.
    2.Requires large datasets to train model to make it efficient.

3.RawNet2 (CNN based model):
  * Key Technical Innovations:
    1.Uses raw audio waveforms directly with CNN based model.

  * Reported Performance Metrics:
    1. EER ~ 3.5%.
    2. Strong generalization across different datasets.

  * Significance of model:
    1.Computationally efficient comapred to transformers.
    2.Works directly on raw audio.

  * Limitations:
    1.Complex to understand than transformer models.
    2.Struggle with low quality audio.


Implementation Process:
 Challenges Encountered & Solutions:
  1.Differentiating High-Quality AI-Generated Speech from Real Human Speech:- One of the most significant challenges was distinguishing high-quality AI-generated speech from real human speech, particularly when 
  the AI-generated voices were trained on large datasets and enhanced with prosody modeling to introduce natural variations in speech. These models often remove traditional artifacts that make synthetic speech 
  identifiable.
  Solutions:
    1.Used Wave2Vec2, which processes raw audio waveforms rather than traditional handcrafted features like MFCCs (Mel-Frequency Cepstral Coefficients).
    2.Fine-tuned the model on recent AI-generated speech datasets, including deepfake samples from state-of-the-art text-to-speech (TTS) models.
    3.Introduced spectral analysis augmentation to expose minor distortions that AI-generated voices often introduce.

  2.Imbalanced Dataset (More Real Speech Samples Than AI-Generated):- Publicly available speech datasets contain a significantly larger proportion of real human speech compared to AI-generated samples. This 
  imbalance causes bias in the model, where it learns to classify most samples as "real speech."
  Solutions:
    1.Used data augmentation techniques, such as adding noise, pitch shifts, and speed variations, to increase the diversity of AI-generated speech samples.
    2.mplemented synthetic oversampling (SMOTE) to balance the dataset.
    3.Applied weighted loss functions to penalize misclassification of AI-generated speech more heavily.

  3.Computational Complexity and Real-Time Processing Constraints:- Wave2Vec2 is a large transformer-based model, making it computationally expensive. For real-world applications, the system must operate in real- 
  time without causing significant processing delays.
  Solutions:
    1.Optimized inference by using quantization techniques such as FP16 precision and model pruning.
    2.Converted the trained model to ONNX format for efficient execution on low-power devices.
    3.Used batch processing and dynamic padding to speed up inference.

 Assumptions made:
   1.AI-generated speech has subtle artifacts in frequency and prosody that can be learned by deep learning models.
   2.The dataset is diverse enough to represent real-world scenarios, including multiple languages, accents, and noise conditions.
   3.Fine-tuning pretrained Wave2Vec2 on AI-generated datasets will yield higher accuracy than training a model from scratch.

 Model Selection and Analysis:
   Among various speech detection models Wave2Vec2 was choosen by me due to its unique and efficient technical advantages:
    1.Unlike traditional approaches that rely on feature extraction (e.g., spectrograms or MFCCs), Wave2Vec2 processes raw waveforms, making it more effective at detecting subtle distortions in AI-generated 
    speech.
    2.The model is pretrained on massive datasets of real speech, allowing it to learn deep audio representations. Fine-tuning on AI-generated speech further enhances its ability to distinguish real vs. fake 
    audio.
    3.Unlike convolutional networks, transformers capture long-range dependencies in speech, improving detection accuracy in real-world conversations.
    4.With model optimizations, the inference speed is ~50ms per sample, making it suitable for live applications.

 Model Working:
  The first stage of the model is a feature encoder, which consists of multiple 1D convolutional layers that transform the raw waveform into a sequence of lower-dimensional speech representations. These 
  convolutional layers act like a filter bank, capturing important frequency and time-domain characteristics of speech while discarding unnecessary noise.
  Once the feature encoder extracts basic audio representations, these representations are passed into a contextual transformer encoder. This is the core innovation of Wave2Vec2, as it applies self-attention 
  mechanisms to analyze long-range dependencies in speech. The transformer learns to model relationships between different speech segments, allowing it to capture variations in intonation, pitch, and 
  articulation —features that distinguish real human speech from AI-generated speech.
  To train the model, the self-supervised learning approach masks certain portions of the audio signal and asks the model to predict the missing parts. This technique forces the model to develop a deeper 
  understanding of speech structures, making it highly robust to variations in speech patterns.
  For the task of AI-generated speech detection, we fine-tune the pretrained Wave2Vec2 model on a dataset containing both real human speech and AI-generated speech samples. The final classification layer 
  (usually a fully connected layer with a softmax activation) determines whether the input speech belongs to a human or an AI model.
  Since Wave2Vec2 operates directly on raw waveforms, it can detect subtle inconsistencies in AI-generated voices—such as unnatural pitch modulation, synthetic harmonics, or timing irregularities—that might be 
  overlooked by traditional feature-based models. Moreover, because the transformer encoder captures contextual information, the model can analyze entire sentences rather than individual phonemes, making it more 
  accurate in detecting deepfake speech in real conversations rather than isolated words.

 Strength and Weakness:
  1.Strength:
   # Highly accurate detection of AI-generated speech.
   # Works across different AI models, including VALL-E, DeepfakeTTS, and WaveNet.
   # Fast inference allows real-time deployment.
  2.Weakness:
   # May struggle with extremely high-quality synthetic voices that lack traditional artifacts.
   # Requires high end devices for proper functionality.
   # Computationally expensive for low-power edge devices.

 Future Improvements:
  1.Enhancing dataset diversity with new AI speech synthesis models.
  2.Implementing distillation to reduce model size.

 Questions to address:
  1.What were the most significant challenges in implementing this model?
  ->One of the biggest challenges was ensuring the correct preprocessing of raw audio data. Since the model expects a specific sample rate (16kHz), all audio files had to be resampled correctly. Another 
  challenge was handling memory and computational efficiency, as Wave2Vec2 requires significant GPU resources for training. Fine-tuning on a dataset with both human and AI-generated speech required careful 
  hyperparameter tuning to achieve the best performance without overfitting.

  2.How might this approach perform in real-world conditions vs. research datasets?
  ->In a controlled research setting, the model performs exceptionally well due to clean and balanced datasets. However, in real-world scenarios, background noise, audio compression, and speaker variations can 
  affect performance. AI-generated speech is also constantly evolving, meaning models trained on current deepfake voices may struggle with newer, more sophisticated speech synthesis techniques. Continuous 
  retraining on updated datasets would be necessary.

  3.What additional data or resources would improve performance?
  ->More diverse datasets containing different AI-generated voices would improve generalization. Currently, models are trained on existing deepfake speech datasets, but newer AI models generate increasingly 
  realistic voices. Including real-world samples from podcasts, phone calls, and voice assistants would help. Using larger transformer architectures or ensemble learning with multiple detection models could also 
  enhance accuracy.

  4.How would you approach deploying this model in a production environment?
  ->For real-time detection, the model needs to be optimized for low-latency inference. Techniques like quantization (reducing model size) and pruning (removing unnecessary parameters) can make it efficient for 
  deployment on edge devices. A streaming API could be implemented, where incoming audio is processed in small chunks, allowing continuous monitoring. Additionally, periodic retraining with new AI-generated 
  speech samples would ensure the model stays effective against evolving deepfake technologies.



  

