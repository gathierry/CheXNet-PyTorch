# CheXNet-PyTorch

This repository reimplements [CheXNet](https://arxiv.org/abs/1711.05225) in PyTorch. At first, the training, validation and inference are based on the default data splitting provided in the [dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC). It took me quite a while to achieve a similar AUC score as the paper until I split the data in the same way as [arnowang's work](https://github.com/arnoweng/CheXNet) according to the paper. The experiments shows that the data splitting has a great impact on reproducing the result of the paper. There will be a discussion about it in the following part.

## Get Started

### Dependency

- Python3.5
- Numpy
- Pandas
- PyTorch
- cv2
- scikit-learn
- py3nvml and nvidia-ml-py3
- tqdm

### Data Preparation

The [ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) comprises 112120 frontal-view chest X-ray images of 30805 patients with 14 disease labels.
data structure:

	DATA_DIR/
		   |-- images/
		   |    |-- <patient_id>_<number>.png
		   |    |-- ...
		   |-- Data_Entry_2017.csv
		   |-- train_val_list.txt
		   |-- test_list.txt
		   |-- ...
		   
### Training and Inference

Simply run ```python3 trainval.py``` for training and ```python3 test.py``` for inference after setting the configuration in ```config.py```.

## Results

| Pathology     | CheXNet Paper| Retrain (CXR default data split) | Retrain (CheXNet paper data split) |
| ------------- |:------------:|:-------------:|:-------------:|
| Atelectasis   | 0.8094       | 0.7761        |0.8250|
| Cardiomegaly  | 0.9248       | 0.8762        |0.9104|
| Effusion      | 0.8638       | 0.8264        |0.8843|
| Infiltration  | 0.7345       | 0.7008        |0.7130|
| Mass          | 0.8676       | 0.8244        |0.8604|
| Nodule        | 0.7802       | 0.7606        |0.7743|
| Pneumonia     | 0.7680       | 0.7279        |0.7790|
| Pneumothorax  | 0.8887       | 0.8565        |0.8759|
| Consolidation | 0.7901       | 0.7588        |0.8084|
| Edema         | 0.8878       | 0.8461        |0.8954|
| Emphysema     | 0.9371       | 0.9137        |0.9262|
| Fibrosis      | 0.8047       | 0.8164        |0.8394|
| P.T.          | 0.8062       | 0.7762        |0.7805|
| Hernia        | 0.9164       | 0.9299        |0.9287|

## Discussion on Data Splitting

The results differ from each other (0.05 for AUC) with distinct data splitting.

The training sets are almost same. But the test set of CheXNet Paper is smaller than the CXR default one.

|            | CXR Default | CheXNet Paper |
| ---------- |:-----------:|:-------------:|
| Train      | 77988       | 78468         |
| Validation | 8536        | 11219         |
| Test       | 25596       | 22433         |

For each disease, the sample ratio between trainig set and test set is listed below.

| Pathology     | CXR Default Train | CXR Default Test | CheXNet Paper Train | CheXNet Paper Test |
| ------------- |:----------:|:---------:|:-----:|:--------:|
| Atelectasis   | 7405       | 3279      | 8014  | 2423  |
| Cardiomegaly  | 1549       | 1069      | 1954  | 582   |
| Consolidation | 2584       | 1815      | 3263  | 957   |
| Edema         | 1253       | 925       | 1690  | 413   |
| Effusion      | 7852       | 4658      | 9268  | 2756  |
| Emphysema     | 1317       | 1093      | 1799  | 509   |
| Fibrosis      | 1111       | 435       | 1158  | 362   |
| Hernia        | 122        | 86        | 144   | 42    |
| Infiltration  | 12480      | 6112      | 13930 | 3944  |
| Mass          | 3646       | 1748      | 4013  | 1139  |
| Nodule        | 4247       | 1623      | 4381  | 1337  |
| P.T.          | 1997       | 1143      | 2279  | 734   |
| Pneumonia     | 782        | 555       | 1031  | 260   |
| Pneumothorax  | 2394       | 2665      | 3708  | 1089  |
| No Finding    | 45485      | 9861      | 42363 | 11923 |

For CXR default data, the ```training set : test set``` ratio is around ```2:1``` whereas the ratio is over ```3:1``` for the data splitting used in the paper. This could be a reason for the difference observed.

Therefore, data splitting is a key factor for reproducing the paper results.