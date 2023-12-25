# Resume-Classifier
Utilizes ResNet-34 for document classification 

Dataset
Reference: A. W. Harley, A. Ufkes, K. G. Derpanis, "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval," in ICDAR, 2015
For the classifier, I used a sample of the RVL-CDIP dataset. The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) originally contains 16 classes with 25,000 images per class. The subset of the dataset that I used contained only 3 classes – email, resume and scientific publication. Each class had 55 images of documents in grayscale and .png format so the total dataset had 165 images.

Model Architecture:
For the document classification I used ResNet-34. This is a 34 layered convolutional neural network that is well suited for image classification tasks. It is pre-trained on the ImageNet dataset which has 100,000+ images and it is available in the fastAI library. I chose this architecture because it produces a good accuracy without being too computationally expensive like ResNet-50 or ResNet-101 would be. ResNet-34’s characteristics allow it to capture complex patterns in images to accurately classify the inputs.

Training strategy:
I applied data augmentation techniques like rotation, width and height shifts, shear, zoom, and horizontal flip to augment each original image 5 times therefore increasing the dataset size and model robustness. Then I resized each image to a consistent target size of 128x128 and normalized the pixel values, so they are in a range of 0 to 1. After this preprocessing I moved to fine-tuning, in which the pre-trained CNN is adapted for a smaller, specific task. I used CNN learner from fastAI to make a CNN with ResNet34 architecture, then I used the fine-tuning method so the weights are adjusted based on my dataset so it can learn task-specific features for resume classification. I adjusted the hyperparameters like the number of epochs from 4 to 6 which led to an increase in accuracy.

Evaluation metrics:
The model gave an accuracy of 72%. Over the six epochs, the accuracy steadily  increased from 41.4% to 72.4%. From the evaluation metrics we can see the model excels as classifying resumes and scientific publications, with recall score of 0.86 and 0.85 but it struggles with emails giving us a recall of only 0.44. When interpreting the confusion matrix, the diagonal denotes true positives which are correct predictions and is the highest for scientific publications. FP is low in our confusion matrix suggesting low misclassifications. False negatives represent instances where the model fails to identify a document that belongs to a particular category. The elevated number of FN in email suggests the model fails to identify emails correctly.

Strengths and Limitations:
The model’s strengths include robustness provided by data augmentation. Since it is pretrained on another dataset, transfer learning will allow this CNN to perform well even on my limited dataset. This model leverages fastAI which reduces code complexity and provides easy to use features for model analysis. Limitations of the model may include the small dataset which can affect generalization power of the model. Due to the complexity of ResNet-34 architecture the training time and resource requirements may be impacted, so adoption of a simpler architecture could be considered for this resume classification task.
