import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("C:/Users/Sravan Apuri/Desktop/train.csv")
print(train.head(10))
plt.hist(train["Age"])
plt.show()