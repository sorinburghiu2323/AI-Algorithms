import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt

MAX_DEPTH = 10  # Maximum depth the decision tree can reach
TEST_SIZE = 0.2  # Test size percentage value for the dataset

# Load dataset.
dataset = pd.read_csv("data.csv")
dataset.head()

# Assign feature and target columns.
feature_columns = ["Roads:number_intersections", "Roads:diversity", "Roads:total",
                   "Buildings:diversity", "Buildings:total", "LandUse:Mix", "TrafficPoints:crossing",
                   "poisAreas:area_park", "poisAreas:area_pitch", "pois:diversity", "pois:total",
                   "ThirdPlaces:oa_count", "ThirdPlaces:edt_count", "ThirdPlaces:out_count",
                   "ThirdPlaces:cv_count", "ThirdPlaces:diversity", "ThirdPlaces:total", "vertical_density",
                   "buildings_age", "buildings_age:diversity"]

# Preprocess data to encode labels.
le = preprocessing.LabelEncoder()
most_recent_age = le.fit_transform(list(dataset["most_present_age"]))
feature_list = []
for column in feature_columns:
    label = le.fit_transform(list(dataset[column]))
    feature_list.append(label)

x = list(zip(*feature_list))  # Features
y = list(most_recent_age)  # Target

# Split data into train and test fragments.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=1)

# Build decision tree classifier.
clf = DecisionTreeClassifier(max_depth=MAX_DEPTH)
clf = clf.fit(x_train, y_train)

# Predict response for test dataset.
y_pred = clf.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Produce bar chart for feature importance.
fi = clf.feature_importances_
plt.bar([x for x in feature_columns], fi)
plt.xticks(rotation='vertical')
plt.show()
