import argparse
import os
import sys
import platform
from pathlib import Path
import numpy as np 
import pandas as pd
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
current_path = Path(__file__).resolve().parent
sys.path.append(str(current_path))
from model import create_vit_classifier
from data import build_decoder,build_dataset



DATA_PATH=os.path.join(current_path,'data')
BATCH_SIZE=8

def test(model,checkpoint_filepath,valid_dataset,valid_labels,label_cols,fold): 
    model.load_weights(checkpoint_filepath)
    pred=model.predict(valid_dataset)
    predicted_class_indices=np.argmax(pred,axis=1)
    labels=np.argmax(valid_labels,axis=1)
    print("Accuracy : ",accuracy_score(labels,predicted_class_indices))
    print("Classification report")
    print(classification_report(labels,predicted_class_indices))
    a=confusion_matrix(labels,predicted_class_indices)
    array = a.astype(int)
    df_cm = pd.DataFrame(array, index = [i for i in label_cols],
                columns = [i for i in label_cols])
    plt.figure(figsize = (10,10))
    sn.heatmap(df_cm.round(5), annot=True,cmap="Blues",fmt='g')
    save_path=os.path.join(current_path,'models','model_fold_'+str(fold)+'_confusion_matrix.png')
    plt.savefig(save_path)  
    print("Confusion matrix saved : ",save_path)
    #plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,help="model weights")
    ap.add_argument("-c", "--classification", required=True,choices=['binary', 'multiclass',],
      help="classification mode")
    ap.add_argument("-f", "--fold", required=True,  help="fold")
    args = vars(ap.parse_args())
    checkpoint_filepath=args['model']
    classification=args['classification']
    fold=int(args['fold'])
        
    path_train=os.path.join(DATA_PATH,'train.csv')
    df=pd.read_csv(path_train)
    if classification=='binary':
            num_class=2
            df=df[ (df['Class']=='COVID-19') | (df['Class']=='NORMAL')]
            label_cols=['COVID-19','NORMAL']    
        
    if classification=='multiclass':
            num_class=3
            label_cols=['ABNORMAL','COVID-19','NORMAL']
    
    system=platform.system()
    if system=='Windows':
      valid_paths = DATA_PATH + '\\' + df[df['fold'] == fold]['Path']  
    else:
      valid_paths = DATA_PATH + '/' + df[df['fold'] == fold]['Path'] 
    
    valid_labels = df[df['fold'] == fold][label_cols].values
    decoder = build_decoder(with_labels=True, target_size=(500, 500))
    valid_dataset = build_dataset(
                valid_paths, valid_labels, bsize=BATCH_SIZE, decode_fn=decoder,
                repeat=False, shuffle=False, augment=False
            )
    vit_classifier = create_vit_classifier(num_class)
    test(vit_classifier,checkpoint_filepath,valid_dataset,valid_labels,label_cols,fold)