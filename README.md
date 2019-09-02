# Speaker_Verification
Pytorch implementation of dvector for speaker verification

### Explanation
- This code is the implementation of generalized end-to-end loss for speaker verification (https://arxiv.org/abs/1710.10467)
- This paper improved the previous work (End-to-End Text-Dependent Speaker Verification, https://arxiv.org/abs/1509.08062)
- Data augmentation found in this paper (Robust DNN embeddings for speaker recognition, https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)

### Speaker Verification
- Speaker verification task is 1-1 check for the specific enrolled voice and the new voice. This task needs higher accuracy than speaker identification which is N-1 check for N enrolled voices and a new voice. 
- There are two types of speaker verification. 1) Text dependent speaker verification (TD-SV). 2) Text independent speaker verification (TI-SV). The former uses text specific utterances for enrollment and verification, whereas the latter uses text independent utterances.
- Each forward step in this paper, similarity matrix of utterances are calculated and the integrated loss is used for objective function. (Section 2.1)


### Files
- data_preprocess.py  
Extract noise and perform STFT for raw audio. For each raw audio, voice activity detection is performed by using librosa library.

- utils.py   
Containing various functions for training and test.  

- grid_search.py  
grid search for fine tunning

### Usages
1. python data_preprocess.py /data/*/*.wav
   - Explain: for example, one wav file for speaker Tom, /data/Tom/tom_voice1.wav
2. Look at the gridsearch.py script and change the parameters.
3. python gridsearch.py
4. tensorboard --logdir log

### Require
1. pytorch (0.4, 1.0, 1.1)
2. tensorboard
3. librosa

### TODOs
- [x] Add tensorboard support.
- [x] Add accuracy for training and testing part.
- [x] Add VAD for preprocessing.
- [x] Transfer learning.
- [x] Convert wav.file to vector.
- [ ] Add different data augmentation in preprocessing.


### Results
- TI-SV
   - Hidden nodes: 32, layers: 3, dvector length: 512, 9% EER for VCTK.
   - Hidden nodes: 32, layers: 3, dvector length: 256, 3% EER for jnas.
