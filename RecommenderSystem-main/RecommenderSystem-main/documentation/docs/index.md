## Recommender System

CarettaSVD is a SVD based product recommendation system based on the similarities between users. Rather than using product-based similarity, user-based similarity preferred to achieve our goal. 

<hr>

### Data Preprocessing

The data is formatted in ```prepare_data``` method. 

1. Unnecessary columns such as "Transaction No, Date, Product Name and Country" are dropped and a new column named "Total Amount", which is the multiplication of Price and Quantity columns, is added to the dataframe. </br>
2. Same product purchases of each user are grouped together.
3. Also, copy of the original data is splitted into two parts in which 20% of selected user's purchases removed and 80% of them remains the exactly same in order to measure success of the recommendation system. 

</br>

After formating process, user x product matrix is created in ```create_ratings_matrix``` method.

1. An empty matrix is created with the shape of len(unique users) x len(unique products).
2. Product and Customer dictionaries are created. Thanks to these dictionaries we can hold customer and product numbers enumaretad and ordered. (Example: some user numbers in ascending order: 13790, 13793, 13820... Since the increase in customer numbers are not by 1, we map them to 1, 2, and 3...)  
3. At the latest step, the matrix is filled while iterating over customers and products arrays.

<hr>

### Matrix Factorization with SVD

#### SVD

SVD algorithm is one of the most powerful techniques in matrix factorization. The specific feature that distinguishes it from other factorization techniques is SVD can be applicable to any matrix. It reveals the hidden features and correlations between matrix’s columns or rows, which represent the users or the products in our case. 

<hr>

### Appendices

#### Cosine Similarity
*In data analysis, cosine similarity is a measure of similarity between two sequences of numbers. For defining it, the sequences are viewed as vectors in an inner product space, and the cosine similarity is defined as the cosine of the angle between them, that is, the dot product of the vectors divided by the product of their lengths.* </br>
*Given two vectors of attributes, A and B, the cosine similarity, cos(θ), is represented using a dot product and magnitude as*

![cosine_similarity](https://github.com/CARETTA-LAB/RecommenderSystem/blob/main/documentation/img/cosine_similarity.png)

*where and are components of vector and respectively.* [1](https://en.wikipedia.org/wiki/Cosine_similarity)

#### Singular Value Decomposition
*In linear algebra, the singular value decomposition (SVD) is a factorization of a real or complex matrix. It generalizes the eigen decomposition of a square normal matrix with an orthonormal eigen basis to any matrix.* [2](https://en.wikipedia.org/wiki/Singular_value_decomposition) </br>

![svd](https://github.com/CARETTA-LAB/RecommenderSystem/blob/main/documentation/img/svd.png)

A mathematical approach to SVD can be found [here](https://medium.com/intuition/singular-value-decomposition-svd-working-example-c2b6135673b5).



