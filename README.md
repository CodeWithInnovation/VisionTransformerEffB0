
# A novel hybrid Vision transformer-CNN for COVID-19 detection from ECG images

The official tensorflow implementation of the paper : A novel hybrid Vision transformer-CNN for COVID-19 detection from ECG images

## Installation
1. Clone the repository
```bash
  git clone https://github.com/CodeWithInnovation/VisionTransformerEffB0.git
```

2. You can install the dependency using the requirements.txt file in a Python>=3.8.0 environment
```bash
cd VisionTransformerEffB0
pip install -r requirements.txt  # install
```

## Train VisionTransformer-EfficientNetB0
Train using the following command

1. Binary 
```bash
python train.py -c binary 
```

2. Multiclass
```bash
python train.py -c multiclass 
```

## Performances
Test the model
```bash
# Fold 1 , binary classification
python test.py -m models/model_fold_1.h5 -c binary -f 1
```

## References

-  Khan, Ali Haider; Hussain, Muzammil  (2020), “ECG Images dataset of Cardiac and COVID-19 Patients”, Mendeley Data, V1, doi: 10.17632/gwbz3fsgp8.1

- Dataset license : [CC BY 4.0 ](https://creativecommons.org/licenses/by/4.0/)

# Important notes

- The provided code is for research use only.