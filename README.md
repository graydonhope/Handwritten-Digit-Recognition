# Machine-Learning-Neural-Networks
Multi-Class Classification and Neural Networks (multi-class classification)

Building a multi-class classifier using one-vs-all logistic regression models. 

Vectorized the cost function and gradient for faster computations and available scalability for larger datasets. Therefore, the code for this computation does not contain any "loops" and is completely iterative free.

Formula for Regularized Logistic Regression:
![image](https://user-images.githubusercontent.com/41659296/53684492-f8221b80-3cdb-11e9-9554-22331f4a1f1e.png)

This is done by using a matrix product and transpose:
![image](https://user-images.githubusercontent.com/41659296/53684509-5bac4900-3cdc-11e9-946f-2c6a40c84e15.png)



Steps to reach the fully vectorized gradient:
![image](https://user-images.githubusercontent.com/41659296/53684554-f4db5f80-3cdc-11e9-8c1b-fd3923c927f0.png)

to simplify:
![image](https://user-images.githubusercontent.com/41659296/53684568-1dfbf000-3cdd-11e9-9bba-2417fe6e2822.png)



In this section, I implement the feedforward propagation of the neural network.
![image](https://user-images.githubusercontent.com/41659296/53684603-70d5a780-3cdd-11e9-83ea-7b45e6561f98.png)
