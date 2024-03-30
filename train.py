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
from test import test
from tensorflow import keras
import tensorflow_addons as tfa
from keras.callbacks import ModelCheckpoint
current_path = Path(__file__).resolve().parent
sys.path.append(str(current_path))
from model import create_vit_classifier
from data import auto_select_accelerator,build_decoder,build_dataset


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classification", required=True,choices=['binary', 'multiclass',],
	help="classification mode")
args = vars(ap.parse_args())
classification=args['classification']

strategy = auto_select_accelerator()

BATCH_SIZE = strategy.num_replicas_in_sync * 8
DATA_PATH=os.path.join(current_path,'data')
SAVE_PATH=os.path.join(current_path,'models')
LEARNING_RATE= 0.001
WEIGHT_DECAY = 0.0001

# if TPU used ; provide dataset gcs path
#DATA_PATH = 'gs://'


def run_experiment(model,fold):
    optimizer = tfa.optimizers.NovoGrad(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),

        ],
    )
    #print(model.summary())

    weights_path = os.path.join(SAVE_PATH,'model_fold_'+str(fold)+'.h5')
    checkpoint = ModelCheckpoint(weights_path,save_weights_only=True,monitor='val_loss', verbose=1, save_best_only=True, mode='min',)
    callbacks_list = [checkpoint]

    steps_per_epoch = train_paths.shape[0] // BATCH_SIZE
    
    history = model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        callbacks=callbacks_list,
    )

    return history



if __name__ == '__main__':
    path_train=os.path.join(DATA_PATH,'train.csv')
    df=pd.read_csv(path_train)
    
    if classification=='binary':
        NUM_EPOCHS = 10
        num_class=2
        df=df[ (df['Class']=='COVID-19') | (df['Class']=='NORMAL')]
        label_cols=['COVID-19','NORMAL']    
    
    if classification=='multiclass':
        NUM_EPOCHS = 200
        num_class=3
        label_cols=['ABNORMAL','COVID-19','NORMAL']

    for i in range(10):
        print("Train Fold ",i)
        system=platform.system()
        if system=='Windows':
          train_paths = DATA_PATH + '\\' + df[df['fold'] != i]['Path']  #"/train/"
          valid_paths = DATA_PATH + '\\' + df[df['fold'] == i]['Path']  #"/val/"
        else:
          train_paths = DATA_PATH + '/' + df[df['fold'] != i]['Path']  #"/train/"
          valid_paths = DATA_PATH + '/' + df[df['fold'] == i]['Path']  #"/val/"
        train_labels = df[df['fold'] != i][label_cols].values
        valid_labels = df[df['fold'] == i][label_cols].values

        IMSIZE = (224, 240, 260, 300, 380, 456, 528, 600,500)
        IMS = 8
        decoder = build_decoder(with_labels=True, target_size=(IMSIZE[IMS], IMSIZE[IMS]))
        test_decoder = build_decoder(with_labels=False, target_size=(IMSIZE[IMS], IMSIZE[IMS]))

        train_dataset = build_dataset(
                train_paths, train_labels, bsize=BATCH_SIZE, decode_fn=decoder,augment=False
            )

        valid_dataset = build_dataset(
                valid_paths, valid_labels, bsize=BATCH_SIZE, decode_fn=decoder,
                repeat=False, shuffle=False, augment=False
            )
        
        vit_classifier = create_vit_classifier(num_class)
        history = run_experiment(vit_classifier,fold=i)
        hist_df = pd.DataFrame(history.history)
        history_path=os.path.join(SAVE_PATH,'history_fold_'+str(i)+'.csv')
        hist_df.to_csv(history_path)
        checkpoint_filepath=os.path.join(SAVE_PATH,'model_fold_'+str(i)+'.h5')
        test(vit_classifier,checkpoint_filepath,valid_dataset,valid_labels,label_cols,i)