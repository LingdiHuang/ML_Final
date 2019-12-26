# ML_Final
Backgroud \n
Nomao is a search engine of places that ranks results according to what you like and what your friends like. In the first step, they need to extract and structuralize local content. So, they collected data from multiple sources from the web and aggregate them properly. And then they need to detect what data refer to the same place.
The raw data they originally collected contains the names of places, phone numbers, address, fax, GPS. And then, comparison techniques are used depending on data type. For string data, 
they use string comparison functions (like levenshtein, trigram, difference, inclusion and equality) and for other values, like GPS, distance is computed. As a consequence, a single example is defined by 118 features, and The ID refers to one pair of comparison. 

Project Outline \n
In this project, five different models(Logistic, Ridge Logistic, Lasso Logistic, Random Forrest, Support Vector Machine) are used to compare their performance on this dataset. We ran 50-100 times for each model, each trainign sample(0.5 learning rate and 0.9 learning rate) and created error rate box plot. At the same time, we saved cross validation errors of lasso, ridge and svm. In order to select the best hyper parameters, we also created cross validation curve to get the hyperparamters which gave the minimum error rate. We compared Lasso and Ridge parameter importance as well as with random forest. In the end, we compared their training time and prediction performance of each model to see if there is trade-off between training efficiency and accuracy. 

