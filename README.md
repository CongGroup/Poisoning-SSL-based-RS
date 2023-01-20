### Seq-poison
Our model for fake user generating.

#### datasets
Pre-processed user-item interaction records obtained by the original data downloaded from [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). 

We use the “5-core” datasets as described in our paper.

#### Run
Create bi-classifier:
  
```
python train_classify.py
```

Now we get the bi-classifier model *{data_name}_bi_classify.pt*.

Train the generator that generates fake users:

```
python main.py
```

Generate poisoning data (the percentage of fake users can be set)：

```
python generate_data.py
```

