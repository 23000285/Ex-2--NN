<H3>Name: VENKATANATHAN P R</H3>
<H3>Register no: 212223240173</H3>
<H3>Date: 05-04-2025</H3>
<H3>Experiment No. 2 </H3>

## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:



<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:

### IMPORT THE LIBRARIES:
```PYTHON
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Define the Perceptron Class

```PYTHON
class Perceptron:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._b = 0.0  # y-intercept
        self._w = None  # weights assigned to input features
        self.misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter=10):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []

        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)
            
    def f(self, x: np.array) -> float:
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array):
        return np.where(self.f(x) >= 0, 1, -1)
```

### Start your main here ,read the iris data set
```PYTHON
df = pd.read_csv('Iris.csv')
```

### Display the first 5 rows
```PYTHON
df.head()
```

### Display information about the dataset
```PYTHON
df.info()
```

### Basic Statistics
```PYTHON
df.describe()
```

### Checking for missing values
```PYTHON
df.isnull().sum()
```

### Count of each species
```PYTHON
df['Species'].value_counts()
```
### Correlation matrix
```PYTHON
correlation = df.corr(numeric_only=True)
correlation
```

### Extract the Label Column
```PYTHON
y=df.iloc[:,4].values
```

### Extract Features
```PYTHON
x=df.iloc[:,4].values
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.set_title("Iris data set")
ax.set_xlabel("Sepal length in width (cm)")
ax.set_ylabel("Sepal width in width (cm)")
ax.set_zlabel("Petal length in width (cm)")
```
### Select only 3 features for 3D plotting
```PYTHON
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm']].values
```
# Create a 3D plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

### Plot each species
```PYTHON
ax.scatter(x[:50,0], x[:50,1], x[:50,2], color='red', marker='o', s=20, edgecolor='red', label='Iris Setosa')
ax.scatter(x[50:100,0], x[50:100,1], x[50:100,2], color='blue', marker='^', s=20, edgecolor='blue', label='Iris Versicolour')
ax.scatter(x[100:150,0], x[100:150,1], x[100:150,2], color='green', marker='x', s=20, edgecolor='green', label='Iris Virginica')
```

### Labels
```PYTHON
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')

plt.legend(loc='upper left')
plt.title('3D Scatter plot of Iris Dataset')
plt.show()
```
### Prepare features and labels
```python
x = df[['SepalLengthCm', 'PetalLengthCm']].values
y = df['Species'].values
```

### Reduce to first 100 samples (Setosa and Versicolor only)
```python
x = x[0:100, 0:2]
y = y[0:100]
```

### Plot Iris Setosa samples
```python
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
```

### Plot Iris Versicolour samples
```python
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='Versicolour')
```

### Labels and legend
```python
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(loc='upper left')
plt.title("Iris Setosa vs Versicolour")
plt.show()
```
### map the labels to a binary integer value
```python
y = np.where(y == 'Iris-setsa', 1, -1)
```

### Plot histogram before standardization
```python
plt.hist(x[:, 0], bins=100)
plt.title("Feature 1 (Sepal Length) before standardization")
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.savefig("./before.png", dpi=300)
plt.show()
```

### Standardization of the input features
```python
x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()
```

### Plot histogram after standardization (optional)
```python
plt.hist(x[:, 0], bins=100)
plt.title("Feature 1 (Sepal Length) after standardization")
plt.xlabel('Standardized Sepal Length')
plt.ylabel('Frequency')
plt.savefig("./after.png", dpi=300)
plt.show()
```
### Split the data
```py
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
```

### Evaluate the model
```py
y_pred = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
```
### Plot the number of errors during each iteration
```py
plt.plot(range(1, len(classifier.misclassified_samples) + 1), classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassifications')
plt.title('Training Errors per Epoch')
plt.show()
```

### Evaluate the model (print accuracy after plot)
```py
y_pred = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
```


# OUTPUT:
![alt text](<Images/Screenshot 2025-04-28 204658.png>)

![alt text](<Images/Screenshot 2025-04-28 204703.png>)

![alt text](<Images/Screenshot 2025-04-28 204707.png>)

![alt text](<Images/Screenshot 2025-04-28 204711.png>)

![alt text](<Images/Screenshot 2025-04-28 204715.png>)

![alt text](<Images/Screenshot 2025-04-28 204719.png>)

![alt text](<Images/Screenshot 2025-04-28 204726.png>)

![alt text](<Images/Screenshot 2025-04-28 204738.png>)

![alt text](<Images/Screenshot 2025-04-28 204750.png>)

![alt text](<Images/Screenshot 2025-04-28 204750.png>)

![alt text](<Images/Screenshot 2025-04-28 204756.png>)

![alt text](<Images/Screenshot 2025-04-28 204802.png>)

![alt text](<Images/Screenshot 2025-04-28 204809.png>)

# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
