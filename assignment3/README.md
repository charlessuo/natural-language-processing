# NLP Assignment 3
	
## Run HMM bi-gram model
	python3 hmm_bigram.py

## Run HMM tri-gram model
	python3 hmm_trigram.py

Both commands will produce results for Viterbi and beam search (k=3) decoding with add-1 smoothing. In addition, it will show suboptimal sequence rates and completely correct sequence rate for both models. 

## Run Bi-LSTM with CRF layer
	python3 bilstm_crf.py

It will print out dev accuracy for this model.