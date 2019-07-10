# K means

K means implementation in Python using numpy library.


## Usage

```python
from kmean import KMeans
import numpy as np

data = np.genfromtxt('data.csv', delimiter=',')
num_cluster = 3
clf = KMeans(num_cluster)
clf.fit(data)
# centroid id of each data point
print(clf.labels_)
# centroids coordianates
print(clf.cluster_centers_)
# prediction to use on another data set
print(clf.predict(data)
```


Please make sure your data is a 2D matrix.

## License
[MIT](https://choosealicense.com/licenses/mit/)
