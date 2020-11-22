# HELPER FUNCTIONS

To read a csv file and convert into numpy array, i used genfromtxt of the numpy package.
For Example:
```
train_data = np.genfromtxt(train_X_file_path, dtype=np.float64, delimiter=',', skip_header=1)
```
For Example:
```
with open('sample_data.csv', 'w') as csv_file:
	writer = csv.writer(csv_file)
    writer.writerows(data)
```# knnProject
