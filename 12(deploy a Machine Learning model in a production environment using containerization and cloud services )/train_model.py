from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

# load data
iris = load_iris()
X = iris.data
y = iris.target

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl saved!")
