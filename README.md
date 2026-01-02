# Sign Language Recognition Using Convolutional Neural Networks

This project investigates the use of convolutional neural networks (CNNs) for classifying American Sign Language (ASL) letters from image data. The work explores how dataset characteristics, preprocessing choices, and model architecture affect classification performance, with the broader goal of improving accessibility through automated sign language recognition.

---

## Motivation
Sign language recognition presents unique challenges due to variability in hand shapes, lighting conditions, image quality, and signer differences. Traditional feature-based methods often struggle with these complexities, whereas CNNs can automatically learn hierarchical visual features directly from image data. This project examines how deep learning approaches can improve robustness and accuracy in static sign language classification.

---

## Data & Preprocessing
Multiple datasets from Kaggle were evaluated during development, including:
- Sign Language MNIST
- Interpret Sign Language with Deep Learning

For the final model, **Sign Language MNIST** was selected due to its balance of accuracy and generalization. Preprocessing steps included:
- Removing motion-based classes (J and Z)
- Converting images to grayscale
- Normalizing pixel values
- Resizing images for consistency
- Applying data augmentation (rotation, zoom, translation) to reduce overfitting and class imbalance

---

## Model Development
Several CNN configurations were explored and compared, varying:
- Number of convolutional layers
- Image resolution
- Batch size
- Number of epochs

The final model used a sequential CNN architecture with:
- Three convolutional layers (32, 64, 128 filters)
- Max pooling between layers
- Dense layers following flattening
- ReLU activations and softmax output
- Adam optimizer with categorical cross-entropy loss

This configuration achieved strong performance while balancing accuracy and computational efficiency.

---

## Results
The final model achieved approximately **99% accuracy**, precision, recall, and F1-score on the test set. While performance on curated datasets was strong, evaluation on external images revealed generalization challenges, highlighting the impact of lighting, background variation, and input quality.

These findings reinforce the importance of diverse training data and careful preprocessing when deploying computer vision models in real-world settings.

---

## Challenges & Lessons Learned
Key challenges encountered included:
- Overfitting on training data
- Sensitivity to image quality and lighting
- Class imbalance across datasets
- Performance degradation when moving from curated datasets to real-world inputs

Data augmentation and dataset expansion helped mitigate these issues, but further work is needed to improve robustness.

---

## Future Work
Potential extensions of this project include:
- Incorporating temporal models (e.g., CNN–LSTM or Transformers) to handle motion-based signs
- Expanding training data to include varied lighting, backgrounds, and perspectives
- Improving real-time inference for deployment in web or mobile applications
- Exploring interpretability techniques such as activation maps and feature visualization

---

## Project Files
- **Notebook:** `sign_language_cnn.ipynb`
- **Report:** `Sign_Language_CNN_Report.pdf`

> **Note:** GitHub’s PDF preview may not display clickable hyperlinks.  
> To access embedded links, please download the PDF directly.

- [Download Project Report (PDF)](https://raw.githubusercontent.com/YOURUSERNAME/REPO_NAME/main/Sign_Language_CNN_Report.pdf)

---

## Context
This project was completed as part of graduate-level coursework in data science and reflects a collaborative effort to evaluate and refine deep learning approaches for image-based sign language recognition.
