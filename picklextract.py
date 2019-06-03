import pickle
import json
if __name__=="__main__":                                                        
	file = open("SCR_Model/wsc285_par_probs.pkl",'rb')                                        
	object_file = pickle.load(file)                                             
	file.close()

with open('SCR_Model/wsc285_par_probs.json', 'w') as outfile:
        json.dump(object_file, outfile)
