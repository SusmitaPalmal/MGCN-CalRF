#Five modalities have been taken here each time for the experiment#
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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
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
seed(1)

from imblearn.over_sampling import SMOTE

def smote_upsample(stacked_feature_train, y_train_rf):
  oversample = SMOTE()
  X, y = oversample.fit_resample(stacked_feature_train, y_train_rf) 
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
list_content = combinations(files_content, 6)
list_cite=combinations(files_cite, 6)

file1 = open("output_6mod.csv.csv","w")



for combo_content,combo__cite in zip(list(list_content), list(list_cite)):

      content1=combo_content[0]
      content2=combo_content[1]
      content3=combo_content[2]
      content4=combo_content[3]
      content5=combo_content[4]
      content6=combo_content[5]
      cite1=combo__cite[0]
      cite2=combo__cite[1]
      cite3=combo__cite[2]
      cite4=combo__cite[3]
      cite5=combo__cite[4]
      cite6=combo__cite[5]
      print("content1===",content1)
      file1.write(str(content1))
      file1.write("\n")
      file1.write(str(content2))
      file1.write("\n")
      file1.write(str(content3))
      file1.write("\n")
      file1.write(str(content4))
      file1.write("\n")
      file1.write(str(content5))
      file1.write("\n")
      file1.write(str(content6))
      file1.write("\n")


      # start = time.perf_counter()
      _cites1 = pd.read_csv( cite1,          
          sep="\t",  # tab-separated
          # sep='	',
          header=None,  # no heading row
          names=["target", "source"],  # set our own names for the columns
      )
      # print(_cites)

      _raw_content1 = pd.read_csv(          
          content1,
          sep=",",  # tab-separated
          #sep="\t", # space-separated
          header=None,  # no heading row
          #names=["id", *_feature_names, "subject"],  # set our own names for the columns
      )

      _cites2 = pd.read_csv( cite2,          
          sep="\t",  # tab-separated
          # sep='	',
          header=None,  # no heading row
          names=["target", "source"],  # set our own names for the columns
      )
      # print(_cites)

      _raw_content2 = pd.read_csv(          
          content2,
          sep=",",  # tab-separated
          #sep="\t", # space-separated
          header=None,  # no heading row
          #names=["id", *_feature_names, "subject"],  # set our own names for the columns
      )

      _cites3 = pd.read_csv( cite3,          
          sep="\t",  # tab-separated
          # sep='	',
          header=None,  # no heading row
          names=["target", "source"],  # set our own names for the columns
      )
      # print(_cites)

      _raw_content3 = pd.read_csv(          
          content3,
          sep=",",  # tab-separated
          #sep="\t", # space-separated
          header=None,  # no heading row
          #names=["id", *_feature_names, "subject"],  # set our own names for the columns
      )

      _cites4= pd.read_csv( cite4,          
          sep="\t",  # tab-separated
          # sep='	',
          header=None,  # no heading row
          names=["target", "source"],  # set our own names for the columns
      )
      # print(_cites)

      _raw_content4 = pd.read_csv(          
          content4,
          sep=",",  # tab-separated
          #sep="\t", # space-separated
          header=None,  # no heading row
          #names=["id", *_feature_names, "subject"],  # set our own names for the columns
      )

      _cites5= pd.read_csv( cite5,          
          sep="\t",  # tab-separated
          # sep='	',
          header=None,  # no heading row
          names=["target", "source"],  # set our own names for the columns
      )
      # print(_cites)

      _raw_content5 = pd.read_csv(          
          content5,
          sep=",",  # tab-separated
          #sep="\t", # space-separated
          header=None,  # no heading row
          #names=["id", *_feature_names, "subject"],  # set our own names for the columns
      )

      _cites6= pd.read_csv( cite6,          
          sep="\t",  # tab-separated
          # sep='	',
          header=None,  # no heading row
          names=["target", "source"],  # set our own names for the columns
      )
      # print(_cites)

      _raw_content6 = pd.read_csv(          
          content6,
          sep=",",  # tab-separated
          #sep="\t", # space-separated
          header=None,  # no heading row
          #names=["id", *_feature_names, "subject"],  # set our own names for the columns
      )



      print("shape modality1=========",_raw_content1.shape)
      _raw_content1.rename(columns={ _raw_content1.columns[0]: "id" }, inplace = True)
      _raw_content1.rename(columns={ _raw_content1.columns[-1]: "subject" }, inplace = True)
      #_raw_content
      _content_str_subject1 = _raw_content1.set_index("id")
      _content_no_subject1 = _content_str_subject1.drop(columns="subject")

      #print("shape is",_content_no_subject)
      _no_subject1 = StellarGraph({"paper": _content_no_subject1}, {"cites": _cites1})
      G1=_no_subject1
      node_label1 = _content_str_subject1["subject"]

      print("shape modality2=========",_raw_content2.shape)
      _raw_content2.rename(columns={ _raw_content2.columns[0]: "id" }, inplace = True)
      _raw_content2.rename(columns={ _raw_content2.columns[-1]: "subject" }, inplace = True)
      #_raw_content
      _content_str_subject2= _raw_content2.set_index("id")
      _content_no_subject2= _content_str_subject2.drop(columns="subject")

      #print("shape is",_content_no_subject)
      _no_subject2 = StellarGraph({"paper": _content_no_subject2}, {"cites": _cites2})
      G2=_no_subject2
      node_label2= _content_str_subject2["subject"]

      print("shape modality3=========",_raw_content3.shape)
      _raw_content3.rename(columns={ _raw_content3.columns[0]: "id" }, inplace = True)
      _raw_content3.rename(columns={ _raw_content3.columns[-1]: "subject" }, inplace = True)
      #_raw_content
      _content_str_subject3= _raw_content3.set_index("id")
      _content_no_subject3= _content_str_subject3.drop(columns="subject")

      #print("shape is",_content_no_subject)
      _no_subject3 = StellarGraph({"paper": _content_no_subject3}, {"cites": _cites3})
      G3=_no_subject3
      node_label3= _content_str_subject3["subject"]

      print("shape modality4=========",_raw_content4.shape)
      _raw_content4.rename(columns={ _raw_content4.columns[0]: "id" }, inplace = True)
      _raw_content4.rename(columns={ _raw_content4.columns[-1]: "subject" }, inplace = True)
      #_raw_content
      _content_str_subject4= _raw_content4.set_index("id")
      _content_no_subject4= _content_str_subject4.drop(columns="subject")

      #print("shape is",_content_no_subject)
      _no_subject4 = StellarGraph({"paper": _content_no_subject4}, {"cites": _cites4})
      G4=_no_subject4
      node_label4= _content_str_subject4["subject"]

      print("shape modality5=========",_raw_content5.shape)
      _raw_content5.rename(columns={ _raw_content5.columns[0]: "id" }, inplace = True)
      _raw_content5.rename(columns={ _raw_content5.columns[-1]: "subject" }, inplace = True)
      #_raw_content
      _content_str_subject5= _raw_content5.set_index("id")
      _content_no_subject5= _content_str_subject5.drop(columns="subject")

      #print("shape is",_content_no_subject)
      _no_subject5 = StellarGraph({"paper": _content_no_subject5}, {"cites": _cites5})
      G5=_no_subject5
      node_label5= _content_str_subject5["subject"]

      print("shape modality6=========",_raw_content6.shape)
      _raw_content6.rename(columns={ _raw_content6.columns[0]: "id" }, inplace = True)
      _raw_content6.rename(columns={ _raw_content6.columns[-1]: "subject" }, inplace = True)
      #_raw_content
      _content_str_subject6= _raw_content6.set_index("id")
      _content_no_subject6= _content_str_subject6.drop(columns="subject")

      #print("shape is",_content_no_subject)
      _no_subject6 = StellarGraph({"paper": _content_no_subject6}, {"cites": _cites6})
      G6=_no_subject6
      node_label6= _content_str_subject6["subject"]





      #==============================================================================================================
      #============== Clinical modality==================================================================================

      df3 = pd.read_csv('file_cln.csv',header = None) 
      array = df3.values
      #array = df3.values
      X3 = array[:,0:-1]
      X3 = preprocessing.scale(X3)
      array1 = df3.values
      y3 = array1[:,-1]
     
      ACC_=0
      MCC_=0
      PRE_=0
      SEN_=0
      SPE_=0
      BALN=0
      F1_=0

      for itr in range (0,1):
            AVG_SENSITIVITY=0
            AVG_SPECIFICITY=0 
            AVG_PRECISION=0
            avg_f1=0
            avg_acc=0
            avgMcc=0
            avgBalAcc=0
            no_of_fold=10
            i=1
            kf=StratifiedKFold(n_splits=no_of_fold, random_state=22, shuffle=True)
            for train_index,test_index in kf.split(_content_no_subject1,node_label1):
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
                  if(content3 !="file_cln.csv"):
                      str1="mod3" 
                      train_embd3, test_embd3=StellerGraphConvolution(train_subjectsMain, train_subjects,test_subjects,val_subjects,G3,node_label3,str1,1,i)
                  else:
                      train_embd3=X3_train
                      test_embd3=X3_test      

                #   for modality4=====
                  if(content4 !="file_cln.csv"):
                      str1="mod4" 
                      train_embd4, test_embd4=StellerGraphConvolution(train_subjectsMain, train_subjects,test_subjects,val_subjects,G4,node_label4,str1,1,i)
                  else:
                      train_embd4=X3_train
                      test_embd4=X3_test

                #   for modality5=====
                  if(content5 !="file_cln.csv"):
                      str1="mod5" 
                      train_embd5, test_embd5=StellerGraphConvolution(train_subjectsMain, train_subjects,test_subjects,val_subjects,G5,node_label5,str1,1,i)
                  else:
                      train_embd5=X3_train
                      test_embd5=X3_test    

                #   for modality6=====
                  if(content6 !="file_cln.csv"):
                      str1="mod6" 
                      train_embd6, test_embd6=StellerGraphConvolution(train_subjectsMain, train_subjects,test_subjects,val_subjects,G6,node_label6,str1,1,i)
                  else:
                      train_embd6=X3_train
                      test_embd6=X3_test         

   

              
                  
                  X_train=np.concatenate((train_embd1,train_embd2,train_embd3,train_embd4,train_embd5,train_embd6), axis=1)
                  X_test=np.concatenate((test_embd1,test_embd2,test_embd3,test_embd4,test_embd5,test_embd6), axis=1)
                  X_train,y3_train=smote_upsample( X_train,y3_train)
             
                  #=====================Calibrated Classifier using Random Fores================================
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
            file1.write("\n")

 
file1.close()
