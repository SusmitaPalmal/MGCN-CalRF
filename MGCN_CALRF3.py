
#Three modalities have been taken here each time for the experiment#
import sys
import os
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from stellargraph import StellarGraph
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import time
import pandas as pd
# from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
#from keras.utils.vis_utils import plot_model
from numpy import savetxt
from sklearn.model_selection import StratifiedKFold
from numpy.random import seed
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sympy import solve, symbols
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve,auc
from sklearn import svm
#from sklearn.naive_bayes import GaussianNB
seed(1)

from imblearn.over_sampling import SMOTE

def smote_upsample(stacked_feature_train, y_train_rf):
  oversample = SMOTE()
  #print("before upsampling shape: \n ")
  #print(y_train_rf)

  X, y = oversample.fit_resample(stacked_feature_train, y_train_rf)
  #print(" after upsampling shape: \n ")
  #print(y)
  return(X,y)







#=============== Graph Based Feature extraction===============================

def StellerGraphConvolution(train_subjectsMain, train_subjects,test_subjects,val_subjects,G,node_label,str1,Modality,fold):
      print(train_subjectsMain.value_counts().to_frame())
      # print(train_subjects.value_counts().to_frame())
      print(train_subjects.value_counts().to_frame())

      print(test_subjects.value_counts().to_frame())
      print(val_subjects.value_counts().to_frame())

      #print(val_subjects)


      target_encoding = preprocessing.LabelBinarizer()

      print(train_subjects)

      train_targets = target_encoding.fit_transform(train_subjects)
      train_targets = to_categorical(train_targets, num_classes=2)
      val_targets = target_encoding.transform(val_subjects)
      val_targets = to_categorical(val_targets, num_classes=2)
      test_targets = target_encoding.transform(test_subjects)
      test_targets = to_categorical(test_targets, num_classes=2)
      #print(test_targets)

      #=================================================
      train_targets_main = target_encoding.fit_transform(train_subjectsMain)
      train_targets_main = to_categorical(train_targets_main, num_classes=2)
      #=================================================

      generator = FullBatchNodeGenerator(G, method="gcn")
      ######
      train_gen_main = generator.flow(train_subjectsMain.index, train_targets_main)
      ######
      train_gen = generator.flow(train_subjects.index, train_targets)
      if Modality==1 :
        gcn = GCN(
            #layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
            # layer_sizes=[200, 100, 50]
            layer_sizes=[300, 200 , 100], activations=["relu", "relu","relu"], generator=generator, dropout=0.5
        )
      elif Modality==2: 
         gcn = GCN(
            #layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
            layer_sizes=[300, 200,150], activations=["relu", "relu","relu"], generator=generator, dropout=0.5
        )   
      elif Modality==3: 
         gcn = GCN(
            #layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
            layer_sizes=[16, 16,16], activations=["relu", "relu","relu"], generator=generator, dropout=0.5
        ) 
      x_inp, x_out = gcn.in_out_tensors()
      print(x_out)

      predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

      model = Model(inputs=x_inp, outputs=predictions)
      #print(x_inp)
      model.compile(
          optimizer=optimizers.Adam(lr=0.01),
          loss=losses.categorical_crossentropy,
          metrics=["acc"],
      )

      val_gen = generator.flow(val_subjects.index, val_targets)


      patience_=10 #20
      es_callback = EarlyStopping(monitor="val_acc", patience=patience_, restore_best_weights=True)

      history = model.fit(
          train_gen,
          epochs=200,
          validation_data=val_gen,
          verbose=2,
          shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
          callbacks=[es_callback],
      )

      sg.utils.plot_history(history)

      test_gen = generator.flow(test_subjects.index, test_targets)
      test_metrics = model.evaluate(test_gen)
    
      test_metrics = model.evaluate(train_gen_main)
      
    

      all_nodes = node_label.index
      all_gen = generator.flow(all_nodes)
      all_predictions = model.predict(all_gen)
      node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())

      node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
      df = pd.DataFrame({"Predicted": node_predictions, "True": node_label})
      
      embedding_model = Model(inputs=x_inp, outputs=x_out)
      train_emb = embedding_model.predict(train_gen_main)
      test_emb = embedding_model.predict(test_gen)
      all_emb= embedding_model.predict(all_gen)
      #print(emb)
      train_result = train_emb[0,:, :]
      test_result= test_emb[0,:, :]
      all_result=all_emb[0,:, :] 
      return train_result, test_result





#=========== MAIN ==========================

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from itertools import combinations



files_content = [ "file_cln.csv",\
        "file_cnv.csv",\
         "file_dna.csv",\
         "file_mir.csv",\
         "file_mrna.csv",\
         "file_wsi.csv"]

files_cite= ["edges/cln0.9_edges.cites",\
           "edges/cnv_edges.cites",\
              "edges/DNA_edges.cites",\
              "edges/mir_edges.cites",\
              "edges/mrna_edges.cites",\
              "edges/wsi_edges.cites"]     

# Get all permutations of [1, 2, 3] 
list_content = combinations(files_content, 3)
list_cite=combinations(files_cite, 3)

file1 = open("output_3mod.csv.csv","w")



for combo_content,combo__cite in zip(list(list_content), list(list_cite)):

      content1=combo_content[0]
      content2=combo_content[1]
      content3=combo_content[2]
      
      cite1=combo__cite[0]
      cite2=combo__cite[1]
      cite3=combo__cite[2]
      
      
      print("content1===",content1)
      file1.write(str(content1))
      file1.write("\n")
      file1.write(str(content2))
      file1.write("\n")
      file1.write(str(content3))
      file1.write("\n")
     

      # start = time.perf_counter()
      DATA_cites1 = pd.read_csv( cite1,          
          sep="\t",  # tab-separated
          # sep='	',
          header=None,  # no heading row
          names=["target", "source"],  # set our own names for the columns
      )
      # print(DATA_cites)

      DATA_raw_content1 = pd.read_csv(          
          content1,
          sep=",",  # tab-separated
          #sep="\t", # space-separated
          header=None,  # no heading row
          #names=["id", *DATA_feature_names, "subject"],  # set our own names for the columns
      )

      DATA_cites2 = pd.read_csv( cite2,          
          sep="\t",  # tab-separated
          # sep='	',
          header=None,  # no heading row
          names=["target", "source"],  # set our own names for the columns
      )
      # print(DATA_cites)

      DATA_raw_content2 = pd.read_csv(          
          content2,
          sep=",",  # tab-separated
          #sep="\t", # space-separated
          header=None,  # no heading row
          #names=["id", *DATA_feature_names, "subject"],  # set our own names for the columns
      )

      DATA_cites3 = pd.read_csv( cite3,          
          sep="\t",  # tab-separated
          # sep='	',
          header=None,  # no heading row
          names=["target", "source"],  # set our own names for the columns
      )
      # print(DATA_cites)

      DATA_raw_content3 = pd.read_csv(          
          content3,
          sep=",",  # tab-separated
          #sep="\t", # space-separated
          header=None,  # no heading row
          #names=["id", *DATA_feature_names, "subject"],  # set our own names for the columns
      )

    


      print("shape modality1=========",DATA_raw_content1.shape)
      DATA_raw_content1.rename(columns={ DATA_raw_content1.columns[0]: "id" }, inplace = True)
      DATA_raw_content1.rename(columns={ DATA_raw_content1.columns[-1]: "subject" }, inplace = True)
      #DATA_raw_content
      DATA_content_str_subject1 = DATA_raw_content1.set_index("id")
      DATA_content_no_subject1 = DATA_content_str_subject1.drop(columns="subject")

      #print("shape is",DATA_content_no_subject)
      DATA_no_subject1 = StellarGraph({"paper": DATA_content_no_subject1}, {"cites": DATA_cites1})
      G1=DATA_no_subject1
      node_label1 = DATA_content_str_subject1["subject"]

      print("shape modality2=========",DATA_raw_content2.shape)
      DATA_raw_content2.rename(columns={ DATA_raw_content2.columns[0]: "id" }, inplace = True)
      DATA_raw_content2.rename(columns={ DATA_raw_content2.columns[-1]: "subject" }, inplace = True)
      #DATA_raw_content
      DATA_content_str_subject2= DATA_raw_content2.set_index("id")
      DATA_content_no_subject2= DATA_content_str_subject2.drop(columns="subject")

      #print("shape is",DATA_content_no_subject)
      DATA_no_subject2 = StellarGraph({"paper": DATA_content_no_subject2}, {"cites": DATA_cites2})
      G2=DATA_no_subject2
      node_label2= DATA_content_str_subject2["subject"]

      print("shape modality3=========",DATA_raw_content3.shape)
      DATA_raw_content3.rename(columns={ DATA_raw_content3.columns[0]: "id" }, inplace = True)
      DATA_raw_content3.rename(columns={ DATA_raw_content3.columns[-1]: "subject" }, inplace = True)
      #DATA_raw_content
      DATA_content_str_subject3= DATA_raw_content3.set_index("id")
      DATA_content_no_subject3= DATA_content_str_subject3.drop(columns="subject")

      #print("shape is",DATA_content_no_subject)
      DATA_no_subject3 = StellarGraph({"paper": DATA_content_no_subject3}, {"cites": DATA_cites3})
      G3=DATA_no_subject3
      node_label3= DATA_content_str_subject3["subject"]



      #==============================================================================================================
      #============== clinical modality==================================================================================

      df3 = pd.read_csv('file_cln.csv',header = None) 
      array = df3.values
      #array = df3.values
      X3 = array[:,0:-1]
      X3 = preprocessing.scale(X3)
      array1 = df3.values
      y3 = array1[:,-1]
      # y3 = 1 - y3




      ACC_=0
      MCC_=0
      PRE_=0
      SEN_=0
      SPE_=0
      BALN=0
      F1_=0
      AUC_=0

      for itr in range (0,1):
            AVG_SENSITIVITY=0
            AVG_SPECIFICITY=0 
            AVG_PRECISION=0
            avg_f1=0
            avg_acc=0
            avgMcc=0
            avgBalAcc=0
            no_of_fold=10
            avgAUC=0
            i=1
            kf=StratifiedKFold(n_splits=no_of_fold, random_state=22, shuffle=True)
            for train_index,test_index in kf.split(DATA_content_no_subject1,node_label1):
                  # print(train_index)
                  # print(test_index)
                  # break
                  print("fold number ################################################",i)
                  X3_train, X3_test = X3[train_index], X3[test_index]
                  y3_train, y3_test = y3[train_index], y3[test_index]

                  # print(type(train_index))
                  train_index=train_index+1
                  test_index=test_index+1

                  # print(node_label)
                  
                  #   break
                  train_subjectsMain, test_subjects = node_label1[train_index], node_label1[test_index]
                  train_subjects, val_subjects = model_selection.train_test_split(train_subjectsMain, test_size=0.10, random_state=20,stratify=train_subjectsMain)
                  
                #   for modality1=====
                  if(content1 !="file_cln.csv"):
                      str1="mod1"
                      train_embd1, test_embd1=StellerGraphConvolution(train_subjectsMain, train_subjects,test_subjects,val_subjects,G1,node_label1,str1,1,i)
                  else:
                      train_embd1=X3_train
                      test_embd1=X3_test

                #   for modality2=====
                  if(content2 !="file_cln.csv"):
                      str1="mod2" 
                      train_embd2, test_embd2=StellerGraphConvolution(train_subjectsMain, train_subjects,test_subjects,val_subjects,G2,node_label2,str1,1,i)
                  else:
                      train_embd2=X3_train
                      test_embd2=X3_test

                #   for modality3=====
                  if(content3 !="/file_cln.csv"):
                      str1="mod3" 
                      train_embd3, test_embd3=StellerGraphConvolution(train_subjectsMain, train_subjects,test_subjects,val_subjects,G3,node_label3,str1,1,i)
                  else:
                      train_embd3=X3_train
                      test_embd3=X3_test      


                
                  
                  X_train=np.concatenate((train_embd1,train_embd2,train_embd3), axis=1)
                  X_test=np.concatenate((test_embd1,test_embd2,test_embd3), axis=1)
                  X_train,y3_train=smote_upsample( X_train,y3_train)
                
                  #====================Calibrated Classifier using Random Forest=================================
                  base_classifiers = []

                  # Number of base classifiers to use
                  num_classifiers = 6

                  # Create and train multiple base classifiers
                  for k1 in range(num_classifiers):
                      clf = RandomForestClassifier(n_estimators=100, random_state=i)
                      clf.fit(X_train, y3_train)
                      
                      # Calibrate the classifier to obtain conformal predictions
                      calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
                      calibrated_clf.fit(X_train, y3_train)
                      
                      base_classifiers.append(calibrated_clf)

                  # Make predictions and obtain conformal predictions
                  conformal_predictions = []
                  for clf in base_classifiers:
                      conformal_predictions.append(clf.predict(X_test))

                  # Combine conformal predictions to make final predictions using the "voting" method
                  final_predictions = np.round(np.mean(conformal_predictions, axis=0))
                  y_pred1=final_predictions
                  
                  #===Make probability predictions and obtain conformal predictions==============
                  conformal_predictions1 = []
                  for clf in base_classifiers:
                        conformal_predictions1.append(clf.predict_proba(X_test)[:, 1])  # Use predicted probabilities for class 1

              
                  final_predictions1 = np.mean(conformal_predictions1, axis=0)     
                  auc_value = roc_auc_score(y3_test, final_predictions1)

                  #=====================================================
                  
                  cm1 = confusion_matrix(y3_test,y_pred1)
                  print('Confusion Matrix : \n', cm1) 

                  TP=cm1[1][1]
                  TN=cm1[0][0]
                  FP= cm1[0][1]
                  FN=cm1[1][0]

                  TPR = TP/(TP+FN) 
                  TNR = TN/(TN+FP) 
                  PPV = TP/(TP+FP) 
                  f1_value=(2*PPV*TPR)/(PPV+TPR) 
                  AVG_SENSITIVITY=AVG_SENSITIVITY+  TPR
                  AVG_SPECIFICITY=AVG_SPECIFICITY+TNR
                  AVG_PRECISION=AVG_PRECISION+ PPV
                  b_ac=(TPR+TNR)/2
                  avgBalAcc=avgBalAcc+b_ac 
                  avg_f1=avg_f1+f1_value  
                  acc= (TP+TN)/(TP+FP+TN+FN)
                  avg_acc=avg_acc+acc
                  mcc=matthews_corrcoef(y3_test,y_pred1)
                  avgMcc=avgMcc+mcc
                  avgAUC=avgAUC+auc_value            
 
                  
                  i=i+1
         
            avg_acc=avg_acc/10
            avg_acc = round(avg_acc, 3)
            avgMcc=avgMcc/10
            avgMcc = round(avgMcc, 3)
            AVG_PRECISION=AVG_PRECISION/10
            AVG_PRECISION = round(AVG_PRECISION, 3)
            AVG_SENSITIVITY=AVG_SENSITIVITY/10
            AVG_SENSITIVITY = round(AVG_SENSITIVITY, 3)
            AVG_SPECIFICITY=AVG_SPECIFICITY/10
            AVG_SPECIFICITY = round(AVG_SPECIFICITY, 3)
            avgBalAcc=avgBalAcc/10
            avgBalAcc = round(avgBalAcc, 3) 
            avg_f1=avg_f1/10
            avg_f1 = round(avg_f1, 3)
            avgAUC=avgAUC/10
            avgAUC = round(avgAUC, 3)            

            file1.write("\n avg Acc ,\t")         
            file1.write(str(avg_acc))  
            file1.write("\n avg mcc ,\t")           
            file1.write(str(avgMcc))
            file1.write("\n avg sensitivity or recall ,\t")
            file1.write(str(AVG_SENSITIVITY))
            file1.write("\n avg specificity ,\t")
            file1.write(str(AVG_SPECIFICITY))
            file1.write("\n avg precision ,\t")
            file1.write(str(AVG_PRECISION))
            file1.write(" \n avg f1 score, \t")
            file1.write(str(avg_f1))
            bal_ac=(AVG_SENSITIVITY+AVG_SPECIFICITY)/2
            file1.write(" \n balanced accuracy, \t")
            file1.write(str(bal_ac))
            file1.write(" \n avg AUC, \t")                        
            file1.write(str(avgAUC))         
            file1.write("\n")

file1.close()
