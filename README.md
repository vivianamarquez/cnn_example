# Classifying "Mitchell vs Everyone Else" Using the CMU Face Dataset 

## Business Goal
This project showcases the potential for machine learning models to automate tasks such as facial recognition and classification in constrained environments. While current performance is high, further development with a more diverse dataset could lead to applications in security, personal identification, and human-computer interaction where distinguishing individuals in varied conditions is critical.

## Context
This project aims to classify images of a person named "Mitchell" against other individuals using a deep learning model. We utilized a subset of the CMU Face Images dataset, consisting of grayscale images resized to 32x32 pixels for model input. The model uses the LeNet-5 architecture, a classic convolutional neural network, to distinguish between "Mitchell" and "Everyone else." The project highlights the challenges of working with imbalanced datasets and explores strategies to improve classification performance using techniques like weighted loss functions.

## DATA
The dataset used in this project comes from the CMU Face Images collection, containing images of 20 different people captured under different conditions (pose, emotion, and eye state). The images were resized from their original resolution of 128x120 to 32x32 pixels to match the input requirements of the LeNet-5 model. The data is heavily imbalanced, with far more images of "Everyone else" than "Mitchell." This imbalance was handled using class weights during training.

Source: [CMU Face Images Dataset](https://archive.ics.uci.edu/dataset/124/cmu+face+images)

## MODEL 
We used the LeNet-5 architecture, originally designed for the MNIST dataset. This model was selected for its simplicity and effectiveness in classifying small grayscale images. The architecture was modified to accommodate binary classification ("Mitchell" vs "Everyone else") by adjusting the final output layer to two classes. Additionally, the Adam optimizer was used instead of the original SGD for better convergence on this specific task. Class weights were applied to handle the data imbalance between the two classes. 

## HYPERPARAMETER OPTIMZATION
The key hyperparameters optimized in this project were:
- Learning rate: Set to 0.001 for the Adam optimizer, which showed faster convergence.
- Batch size: Set to 64, balancing training speed and memory usage.
- Epochs: Training was capped at 100 epochs, with early stopping implemented based on validation loss.
- Class weights: Adjusted to account for the imbalance in the dataset, giving more weight to the minority class ("Mitchell").
These hyperparameters were chosen based on empirical testing and prior knowledge of similar classification problems.

## RESULTS
The model achieved 100% accuracy on the training and test sets, indicating it perfectly learned to classify the dataset. However, this high accuracy raises concerns about overfitting, suggesting the model may not generalize well to new, unseen data. The perfect accuracy likely reflects the simplicity and limited variability of the dataset, making it easy for the model to overfit.

Key learnings:
- The dataset is too small and imbalanced to provide robust generalization.
- Although the model performs well on the given dataset, it may fail when exposed to more complex or varied data.
- Future improvements should include using a larger, more diverse dataset and potentially a more complex architecture to avoid overfitting.

## Conclusions
While the model performs perfectly on the current data, overfitting is a major concern due to the small, imbalanced dataset. This highlights the need for better data and more robust architectures for real-world applications. Future work should focus on improving the model's generalizability by introducing more varied data and refining the architecture.

## Actionable Insights
- Data Collection: Acquire a more diverse dataset with more examples of both "Mitchell" and other individuals to improve model robustness.
- Model Development: Experiment with more complex architectures (e.g., deeper CNNs) to improve generalization and prevent overfitting.
- Deployment Considerations: Given the model’s current limitations, it should not be deployed in environments requiring high generalization without further refinement.
