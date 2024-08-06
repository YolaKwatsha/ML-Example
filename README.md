# Machine Learning

On this example I use the Iris dataset, which includes 150 samples with 4 features each: sepal length, sepal width, petal length, and petal width. 
The dataset is divided into training and testing sets. 
This allows us to train the model on one part of the data and evaluate its performance on another part. 
The k-NN algorithm is used for classification. 
The model is trained on the training data, where it learns to classify based on the nearest neighbors. 
The trained model predicts the species of flowers in the test set. 
The accuracy of the model is calculate and printed, which indicates how well it performs on the test data.

We import libraries for data manipulation (pandas), machine learning (scikit-learn), and model evaluation. 
The Iris dataset is loaded with load_iris(). X contains the features, and y contains the target labels. 
The dataset is split into training and testing sets using train_test_split(). 
30% of the data is used for testing, and random_state=42 ensures that the split is reproducible. 
We create a k-NN classifier with KNeighborsClassifier(n_neighbors=3), where n_neighbors is the number of neighbors to consider for classification. 
The model is trained on the training data using knn.fit(X_train, y_train). 
We use the trained model to predict the classes for the test data with knn.predict(X_test). 
The accuracy of the model is computed using accuracy_score() and printed out, indicating how well the model performs on the test set.
