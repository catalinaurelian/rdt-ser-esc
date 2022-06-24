import numpy as np
import IPython.display as ipd #Allows Audio files to be played directly in the notebook
import librosa #library we will use to analyze sounds
import librosa.display #library module which helps visualize the waveforms
import os
import glob
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time as ti
import sys
import soundfile as sf
import tflite_runtime.interpreter as tflite

sns.set_style("whitegrid")
reduced=0
dformat='channels_last'


pick = sys.argv[1]

def read_signal(wfile):
    data, samplerate = sf.read(wfile)               ## use this ideally for files that have the same sampling rate. Can be used for different sampling rates as well, but model accuracy may be affected.
    #data, samplerate = librosa.load(wfile)           ## using librosa.load() resamples the file, thus increasing processing time by seconds. 
    # if int("2") in np.shape(data):
    #   data=librosa.to_mono(data)
    # print(np.shape(data))
    # data = trim_audio(data, top_db=1000)[0]                 ## very high value --> doesn't apply
    # print(f"Data: {data}\n, Type: {type(data)}\n SampleRate: {samplerate}\n")
    N=np.shape(data)[0]
    
    data=np.reshape(data.T,(1,N)) 
    
    #data=np.reshape(data.T,(1,N))
    # iesirea este in formatul acceptat de functiile RDT 
    return data

def NRDT(signal, w, flag, channels):
#Implementation of RDT based spectral image processor  
# Coyright Radu & Ioana DOGARU - 17 July 2020; 20 Oct. 2020
# Code associated with papers: 

# R. Dogaru and I. Dogaru, "A low complexity solution for epilepsy detection using an improved version 
# of the reaction-diffusion transform," 2017 5th International Symposium on Electrical and Electronics Engineering 
# (ISEEE), Galati, 2017, pp. 1-6.

# I. Dogaru, D. Stan and R. Dogaru, "Compact Isolated Speech Recognition on Raspberry-Pi based on 
# Reaction Diffusion Transform," 2019 6th International Symposium on Electrical and Electronics Engineering 
# (ISEEE), Galati, Romania, 2019, pp. 1-4.

# R. Dogaru and I. Dogaru, "RD-CNN: A Compact and Efficient Convolutional Neural Net for Sound Classification", in 
# Proceedings ISETC, 2020. 

# If you find this code useful in your research, please consider citing the above papers 
#----------------------------------------------------------------------
# Signal is an NP.ARRAY - format [1,N]  
# <channels> is a list of delays  
# 
#----------- channels = 1 2 5 ... 9 
    
    signal=signal.astype('float32') # may help  
    Nsamples=np.size(signal,1)
    delmax=w/4 #  delay should be no more than w/4 (w usually is a power of 2)
    res=np.where(channels<=delmax)
    #print(res)
    channels=channels[res]  # remove channels not satisfying this condition. 
    m=np.shape(channels)[0]
    

    spectrograms=Nsamples//w # The number of spectrograms computed
    # print(f"Spectrogram number: {spectrograms}\n Nr samples: {Nsamples}\n w={w}")
    Samples=spectrograms*w # The number of samples used to compute the spectrograms.The other samples are discarded
    matrix=np.reshape(signal[0,0:Samples],(spectrograms,w)) # each line is one to be submited for computation of spectrogram 
    
    
    spectrum=np.zeros((m,spectrograms))
    for i in range(0,spectrograms):
        values=matrix[i,:] # the whole line 
        for k in range(0,m):
            delay=channels[k]  # delays   
            t=np.array(range(delay,w-delay-1))
            difus=np.abs(values[t-delay]+values[t+delay]-2*values[t])
            if flag==0:
                spectrum[k,i]=np.mean(difus)/4
            elif flag==1:
                spectrum[k,i]=np.mean(difus/(np.abs(values[t-delay])+np.abs(values[t+delay])+2*np.abs(values[t])+1e-12))/4
    return spectrum 


def get_features_nrdt(filename, M, w, flag, prag, chan):
# Implements "spectral image (F3org)" using RDT applied on  M  segments, W window size . 
# Gives errors if the number of windows per segment is smaller than  1 
# Needs tuning of M, w. 
# chan - a list of delays for the "spectral" channels 
# prag - it is usually taken 0 (in special cases larger)
# flag - 0 (normal) / 1 (scaled RDT)
#=========================================================================
  signal=read_signal(filename)
  # signal [1,N] este scalaat -1,1 
  delmax=w/4 #  ne asiguram ca delay-ul maxim nu depaseste w/4 (w este de regula putere a lui 2)
  res=np.where(chan<=delmax)
  #print(res)
  channels=chan[res]
  m=np.shape(chan)[0]
  
  # print('Threshold for sample removal', prag )
  # print('Full length of original signal is : ',np.size(signal))
  t1=ti.time()
  res=np.where(np.abs(signal)>=prag)
  semnal=signal[0,res[1]]
  semnal=np.reshape(semnal.T,(1,np.shape(semnal)[0]))
  Features=np.zeros((M*m))
  # print(M, m)
  Feat_spec=np.zeros((M,m)) 
  Npsgm=np.shape(semnal)[1] // M # The number of samples per each segment
  # print('Nsegm=',Npsgm, 'Windows per each segment: ', Npsgm // w)
  print(f"Number of samples per each segment: {Npsgm}\n Number of windows per each segment: {Npsgm // w}\nWindow size: {w}")
  for isgm in range(0,M): # Calculate the RDT on each segment of the signal 
      ssegment=np.reshape(semnal[0,isgm*Npsgm:(isgm+1)*Npsgm],(1,Npsgm))
      spectrum=NRDT(ssegment,w,flag,chan)
      # print(spectrum)
      mediumRDT=sum(np.transpose(spectrum)) # The medium spectrogram is the sum on columns of the transposed spectrum matrix
      # print(M, mediumRDT)
      Features[isgm*m:(isgm+1)*m]=mediumRDT # The feature vector for the signal to be recognized
      Feat_spec[isgm,:]=mediumRDT.T
  t2=ti.time()
  
  return Features, Feat_spec




if pick=="us_45_6":    ## CAREFUL

    #model = tf.keras.models.load_model('/home/pi/Desktop/catalin/Models_Variables/esc50_test1_tf220.h5')
    x_test = np.load('/home/pi/Desktop/catalin/Models_Variables/us_xtest_fold1_45_6.npy')
    y_test = np.load('/home/pi/Desktop/catalin/Models_Variables/us_ytest_fold1_45_6.npy')
    x_test=x_test.astype("float32")
    test_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test))
    modelstr = 'us_45_6.tflite'
    
elif pick=="esc50_53_6":    ## CAREFUL

    model = tf.keras.models.load_model('/home/pi/Desktop/catalin/Models_Variables/esc50_test1_tf220.h5')
    x_test = np.load('/home/pi/Desktop/catalin/Models_Variables/esc50_savedsplit/esc50_xtest_split_53_6.npy')
    y_test = np.load('/home/pi/Desktop/catalin/Models_Variables/esc50_savedsplit/esc50_ytest_split_53_6.npy')
    x_test=x_test.astype("float32")
    test_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test))
    modelstr = 'esc50_test1_tf220.tflite'

elif pick=="tess_53_6":
    model = tf.keras.models.load_model(f'/home/pi/Desktop/catalin/Models_Variables/tess_test1_tf220.h5')
    x_test = np.load('/home/pi/Desktop/catalin/Models_Variables/Tess_savedsplit/tess_xtest_split.npy')
    y_test = np.load('/home/pi/Desktop/catalin/Models_Variables/Tess_savedsplit/tess_ytest_split.npy')
    x_test=x_test.astype("float32")
    test_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test))
    modelstr = 'tess_test1_tf220.tflite'


elif pick=="rav_53_6":
    model = tf.keras.models.load_model(f'/home/pi/Desktop/catalin/Models_Variables/rav_test1_tf220.h5')
    x_test = np.load('/home/pi/Desktop/catalin/Models_Variables/rav_xtest_split.npy')
    y_test = np.load('/home/pi/Desktop/catalin/Models_Variables/rav_ytest_split.npy')
    x_test=x_test.astype("float32")
    test_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test))
    modelstr = 'rav_test1_tf220.tflite'


elif pick=="esc50_214_7":
    #model = tf.keras.models.load_model(f'/home/pi/Desktop/catalin/Models_Variables/esc_test1_tf220.h5')
    x_test = np.load('/home/pi/Desktop/catalin/Models_Variables/esc_x_test_214_7.npy')
    y_test = np.load('/home/pi/Desktop/catalin/Models_Variables/esc_y_test_214_7.npy')
    x_test=x_test.astype("float32")
    test_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test))
    modelstr = 'esc_214_7.tflite'
    
elif pick=="rte_53_6":
    #model = tf.keras.models.load_model(f'/home/pi/Desktop/catalin/Models_Variables/esc_test1_tf220.h5')
    x_test = np.load('/home/pi/Desktop/catalin/Models_Variables/rte_xtest_split_53_6.npy')
    y_test = np.load('/home/pi/Desktop/catalin/Models_Variables/rte_ytest_split_53_6.npy')
    x_test=x_test.astype("float32")
    test_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test))
    modelstr = 'rte_test_53_6.tflite'
    
elif pick=="comenzi_30_6":
    model = tf.keras.models.load_model(f'/home/pi/Desktop/catalin/Models_Variables/comenzi_test1_tf220.h5')
    #x_test = np.load('/home/pi/Desktop/catalin/Models_Variables/Tess_savedsplit/tess_xtest_split.npy')
    #y_test = np.load('/home/pi/Desktop/catalin/Models_Variables/Tess_savedsplit/tess_ytest_split.npy')
    
def evaluate(modelstr):
    matches=0
    t1=ti.time()
    
    interpreter = tflite.Interpreter(model_path = f'/home/pi/Desktop/catalin/Models_Variables/{modelstr}')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    for i in range(np.shape(x_test)[0]):
        
        shp=np.shape(x_test[i])
        interpreter.set_tensor(input_details[0]['index'], np.reshape(x_test[i],(1,shp[0],shp[1],1)))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_id = np.argmax(output_data)
        
        if (np.argmax(output_data)==np.argmax(y_test[i])):
            matches=matches+1
    score = matches/(np.shape(x_test)[0])*100
    
    t2=ti.time()
    print("Test accuracy:", score, "%", f"(correctly classified {matches} samples out of {len(x_test)} total test samples.")
    print("Time for test set: ", t2-t1)
    print("Latency per input sample: ", 1000*(t2-t1)/np.shape(x_test)[0], 'ms')
    


def recog(test_model, path):
    modelstr=test_model
    Ns=int(sys.argv[1].split("_")[1]) # The sprectrogram dim. is indicated by the name of the model called in argv[1] e.g. esc50_53_6 (dataset_nr-segments_delay-vect-size)
    m=int(sys.argv[1].split("_")[2])  #  -||-
    t1=ti.time()
    my_signal_file=path  
    #print(f"m: {m}")
    
    if m==6:
        chan=np.array([1, 2, 4, 8, 16, 32])
        ws=512
    elif m==7:
        chan=np.array([1, 2, 4, 8, 16, 32, 64])
        ws=1024

    (F3, F3org)=get_features_nrdt(my_signal_file, Ns, ws, 0, 0.0001, chan)
    t2=ti.time()
    print('RDT Processing time: ', t2-t1, ' seconds')
    #plt.imshow(F3org,aspect='auto')
    shp=np.shape(F3org)
    
    interpreter = tflite.Interpreter(model_path = f'/home/pi/Desktop/catalin/Models_Variables/{modelstr}')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    F3org = np.float32(F3org)
    interpreter.set_tensor(input_details[0]['index'], np.reshape(F3org,(1,shp[0],shp[1],1)))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    t3=ti.time()
    predicted_id = np.argmax(output_data)
    print(f"Predicted: {predicted_id}")
    print('Recognition time on R-Pi: \n',t3-t2)
    print('Total recognition time (RDT + Inference): ', t3-t1,' seconds')
  




#print(f"model string: {sys.argv[1]}; path: {sys.argv[2]}")
#evaluate(modelstr)
if len(sys.argv)==3:
    recog(modelstr, sys.argv[2])
else:
    evaluate(modelstr)
    

