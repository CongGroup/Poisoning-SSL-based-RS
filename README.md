### Poisoned_datasets
This is the implementation code for paper 《Poisoning Self-supervised Learning Based Sequential Recommendations》 on ACM SIGIR 2023

For each dataset, we generated fake users whose numbers are 1%, 2%, and 3% of real users.

It is worth noting that, in the three original __Amazon__ datasets (__Beauty__, __Sports and Outdoors__, and __Toys and Games__), none of the users has multiple interactions with the same item, while in the *Yelp* dataset, users often interact with the same item multiple times.
Therefore, to ensure the stealthiness of our attack, when constructing the poisoning data of the __Amazon__ datasets, we let each fake user only interact with the target item once, while in __Yelp__, we allow each fake user to interact with the target item multiple times.
For comparison, we also provide the poisoning data of the __Yelp__ dataset where each user interacts with the same item at most once.

### Seq-poison
Our model for fake user generating.

#### datasets
Original pre-processed user-item interaction records obtained by the data downloaded from [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI) (which is publicly available). 

We use the “5-core” datasets as described in our paper.

#### Run
Create bi-classifier:
  
```
python train_classify.py
```

Now we get the bi-classifier model __{data_name}_bi_classify.pt__.

Train the generator that generates fake users:

```
python main.py
```

Generate poisoning data (the percentage of fake users can be set)：

```
python generate_data.py
```

