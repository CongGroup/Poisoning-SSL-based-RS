# Poisoning-Attack-on-Self-supervised-Learning-Based-Sequential-Recommendation
- ./Original_datasets: obtained by pre-processing the original user interaction data， which is downloaded from [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). After removing the item with too little user interaction and the user with too little item interaction, we obtain this dataset.
- ./Poisoned_datasets: The poisoning data was generated with our poisoning data generator. For each dataset, we generated poisoning data accounting for 1%, 2%, and 3% of the original data.
- ./Seq-poison: Our model for fake sequence generating.
    - ./main.py: the overall architecture of our model, including Part A, Part B, and Part C.
    - ./Bi_classifier model: binary classifier in Part B, which is pre_trained and fixed.
