##Original_datasets
Pre-processed user-item interaction records obtained by the original data downloaded from [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). We use the “5-core” datasets as described in our paper.

##Poisoned_datasets
The poisoning data was generated with our scheme. For each dataset, we generated fake users accounting for 1%, 2%, and 3% of the original data. 

##Seq-poison
Our model for fake sequence generating.
##Run
For simplicity, here we take Beauty as an example.
Create bi-classifier:
``python train_classify.py
Then we get the bi-classifier model *Beauty_bi_classify.pt*.
