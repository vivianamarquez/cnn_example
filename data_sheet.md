# Datasheet 

## Motivation

- For what purpose was the dataset created?

The dataset was created for facial image analysis and recognition tasks, including pose estimation, emotion recognition, and eye status classification.

- Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? Who funded the creation of the dataset?
  The dataset was created by the Robotics Institute at Carnegie Mellon University (CMU). Specific funding information is not provided, but it likely comes from academic or research grants.

 
## Composition

- What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)
  The instances represent grayscale images of 20 different people, captured under various conditions, including different poses, emotions, and eye statuses.
 
- How many instances of each type are there?
  There are over 600 images, with each image representing a combination of different pose, emotion, and eye statuses.

- Is there any missing data?
  There's no missing data given that this was an artificially created dataset.

- Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by    doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)?
  No, the dataset does not contain confidential information. The images appear to have been collected in a controlled research setting with likely participant consent.

## Collection process

- How was the data acquired?
  The images were taken in a lab setting, using controlled conditions where each subject posed for the camera under different variations of facial expressions, head poses, and eye statuses.
  
- If the data is a sample of a larger subset, what was the sampling strategy?
  The dataset is not a sample but a complete set of images collected under specific controlled conditions for each subject.
  
- Over what time frame was the data collected?
  The exact time frame is not specified, but it is assumed that the data was collected in a short-term session for each subject as part of a research project.

## Preprocessing/cleaning/labelling

- Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section.
  The images are stored in .pgm format with filenames that include structured information such as user ID, pose, emotion, eye status, and scale. Additional preprocessing, such as image resizing or scaling, could be done by users of the dataset.
  
- Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?
  The raw images are preserved in their original .pgm format, providing flexibility for users to apply further preprocessing as needed.
 
## Uses

- What other tasks could the dataset be used for?
  The dataset could be used for tasks such as facial recognition, emotion detection, pose estimation, gaze detection, and general computer vision research involving human faces.
  
- Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a dataset consumer might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other risks or harms (e.g., legal risks, financial harms)? If so, please provide a description. Is there anything a dataset consumer could do to mitigate these risks or harms?
  The dataset contains a small and specific subset of the population, which may not be representative of broader populations in terms of gender, ethnicity, or age. Users should be cautious when generalizing results from this dataset to avoid potential biases.
  
- Are there tasks for which the dataset should not be used? If so, please provide a description.
  The dataset should not be used for applications involving sensitive decision-making or real-world deployments that require a diverse representation of human faces.

## Distribution

- How has the dataset already been distributed?
  The dataset is publicly available online through CMU’s website for academic and research purposes.
  
- Is it subject to any copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?
  Yes, the dataset is subject to Carnegie Mellon University's licensing terms and is intended for non-commercial research use.

## Maintenance

- Who maintains the dataset?
  The dataset is maintained by the Robotics Institute at Carnegie Mellon University (CMU). Specific contact details for dataset maintenance are typically available on the CMU dataset webpage.

