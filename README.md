MNIST Autoencoder and Classifier Project

1. Autoencoder
Objective: Encode and decode images into a latent space (~d=12).
Architecture: Convolutional autoencoder with separate encoder, decoder, and MLP models.
Loss Function: Mean L1 error for reconstruction loss.
Results: Reported input vs. reconstructed images and scores.

2. Classifier
Objective: Use encoder architecture from #1 to create a pre-classification feature extractor.
MLP Network: Added a two-layer MLP to classify 10 digit classes.
Training: Used cross-entropy loss. Plotted training/test errors and accuracies.

3. Classifier Decoding
Objective: Use pre-trained classifier encoder from #2 with a fixed decoder network.
Comparison: Compared reconstruction losses with those from #1.
Analysis: Displayed reconstructed digits and analyzed differences in latent embeddings and class variability.

4. Shortage in Training Examples
Objective: Train classifier with only 100 labeled examples.
Method: Used a subset of the MNIST training set.
Results: Reported losses and accuracies over training time; noted any overfitting.

5. Transfer Learning via Fine-tuning
Objective: Fine-tune the pre-trained encoder from #1 using the 100 examples from #4.
Method: Initialized classifier with the pre-trained encoder.
Results: Plotted losses and accuracies over training time and discussed the results.
