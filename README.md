# Spoken Language Identification (LID) Ensemble Engine

An advanced speech classification pipeline that ensembles state-of-the-art Automatic Speech Recognition (ASR) models to perform highly accurate Spoken Language Identification (LID). 

This project extracts and evaluates acoustic features from multiple frontier STT models and aggregates their probability distributions using a Support Vector Machine (SVM) to optimize for multilingual accuracy.

## 🚀 Performance
* **Final Classification Accuracy:** `93.15%`

## 🧠 Architecture & Approach
Relying on a single acoustic model can lead to bottlenecks in diverse multilingual environments. To solve this, this project implements a custom ensemble architecture:
1. **Feature Extraction:** Raw audio waveforms are processed and fed into three distinct, state-of-the-art transformer models via Hugging Face.
2. **Probability Distribution Generation:** Each base model generates predictions/hidden states for the input audio.
3. **SVM Ensemble:** An RBF-kernel Support Vector Machine (SVM) acts as a meta-classifier, aggregating the probability distributions from the underlying STT models to make the final, highly optimized language prediction.

### Base Models Integrated:
* **OpenAI Whisper**
* **Wav2Vec 2.0 (XLSR)**
* **Meta MMS (Massively Multilingual Speech)**

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Frameworks:** PyTorch
* **AI/Audio Libraries:** Hugging Face (Transformers), Librosa
* **Machine Learning:** Scikit-Learn (SVM)
* **Data Processing:** Pandas, NumPy

## 📂 Repository Structure
The pipeline is broken down into modular Jupyter Notebooks for feature extraction, individual model evaluation, and the final ensemble:

* `whisper_svm.ipynb` - Feature extraction and SVM integration specifically for the Whisper model.
* `xlsr.ipynb` - Processing and evaluation pipeline for Wav2Vec 2.0 (XLSR).
* `mms_feedback.ipynb` & `mms_xlsr_probs.ipynb` - Pipeline for Meta MMS and generating combined probabilities with XLSR.
* `svm_probs.ipynb` - Logic for extracting and formatting probability distributions for the meta-classifier.
* `whisper_svm_xlsr.ipynb` - The core ensemble architecture combining Whisper, XLSR, and SVM logic.
* `meta_data.ipynb` - Dataset metadata handling, final ensemble aggregation, and accuracy evaluation yielding 93.15%.

## 💡 Use Case
This architecture is highly relevant for real-time conversational intelligence, global call centers, and GenAI-powered agent-assist suites where dynamic, low-latency multilingual transcription and language routing are critical.