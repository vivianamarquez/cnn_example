# Model Card

## Model Description

**Input:** 
The model takes grayscale images of faces with dimensions 128x120 pixels. These images are resized to 32x32 pixels to match the input requirement of the LeNet-5 architecture. The task is to classify whether the image is of "Mitchell" or "Everyone else."

**Output:**
The model outputs a probability distribution over two classes: "Mitchell" and "Everyone else." The class with the highest probability is chosen as the prediction.

**Model Architecture:** 
The model is a modified version of LeNet-5, with the following changes:
- Input images are resized to 32x32 to match the LeNet-5 requirements.
- The final fully connected layer is modified to output predictions for two classes instead of the original 10.
- The original SGD optimizer is replaced with Adam for better convergence and performance.
- Due to the imbalance in the dataset (with many more "Everyone else" images than "Mitchell" images), class weights were used during training. This ensures that the model does not become biased toward the majority class.

## Performance

While the model achieved perfect accuracy, this is concerning and may indicate overfitting due to the simplicity of the dataset or the architecture's inability to generalize to more complex data.

## Limitations

- Overfitting Risk:
The perfect 100% accuracy is likely a sign of overfitting, especially given the relatively small and imbalanced dataset. The model may not perform well on unseen or more varied data.
- Dataset Bias:
The dataset is highly imbalanced, with significantly fewer "Mitchell" images than "Everyone else" images. This imbalance necessitates careful use of class weights, but the dataset itself may not be diverse enough to represent real-world scenarios.
- Limited Generalization:
Since the model is trained specifically for "Mitchell vs Everyone else," it may not generalize well to other facial classification tasks, especially those involving more complex variations in pose, lighting, and facial expressions.

## Trade-offs

- Accuracy vs. Generalization:
Achieving 100% accuracy suggests that the model fits the training data perfectly, but this comes at the cost of poor generalization to other datasets or more complex scenarios. In future iterations, a more diverse dataset and a more robust architecture could mitigate this issue.
- Model Complexity vs. Dataset:
The LeNet-5 architecture is lightweight and effective for simple tasks like this, but more complex architectures might be necessary for larger, more diverse datasets to prevent overfitting and improve generalization.
- Optimizer Choice:
Replacing SGD with Adam resulted in faster convergence and better performance, but it could contribute to the model overfitting on the current dataset.

## Future Improvements
- Better Dataset:
In the future, a larger and more diverse dataset should be used to improve the model’s robustness and reduce overfitting.
- More Robust Architecture:
A more complex architecture may be required for better generalization, especially if the dataset becomes more diverse and the classification task becomes more nuanced.
