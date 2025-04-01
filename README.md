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
    

