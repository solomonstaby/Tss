debug = False

import pandas as pd
import numpy as np

import json
import pickle

from Trie import VarIden_Trie, Trie_creation

from Text_input_processing import Port_Iden_Remove_TestspecLine, Text_Preprocessing

from LHS_RHS_Classifier import Position_Vectorizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize ,WhitespaceTokenizer

typeclassifier_model_path = r'Processed_Output\models\typeclassifier.sav'

LHS_RHS_model_path = r'Processed_Output\models\LHS_RHS_Classifier.sav'
LHS_RHS_WORDLIST_path = r'Processed_Output\models\LHS_RHS_WORDLIST.json'



def TestSpec_Preprocessing(testspec_step):
    # Generate the Trie using Variable Database 
    MappingLabel_Database_path = 'Database\RQM_to_ECU-TEST_Mapping_CHARCON_V1.xls'

    Interface = Trie_creation(MappingLabel_Database_path)

    # Pre process the Text Data for adjust the symbols
    testspec_step = Text_Preprocessing(testspec_step)

    # Identify and Replace the Variables in the Test spec with keywords using Trie
    testspec_step, Var_Details = VarIden_Trie(testspec_step, Interface)

    # port identification
    testspec_step, port = Port_Iden_Remove_TestspecLine(testspec_step)

    # lower case
    testspec_step = testspec_step.lower()

    return testspec_step, Var_Details, port

def LHS_RHS_Prediction(text):

    # labelled dataset in JSON 
    f = open(LHS_RHS_WORDLIST_path)
    # loading the dataset
    most_freq = json.load(f)

    model = pickle.load(open(LHS_RHS_model_path, 'rb'))

    text_token = word_tokenize(text)

    df = Position_Vectorizer(text_token, most_freq, 'Single')

    label_pred = model.predict(df[most_freq])
    
    return text_token, label_pred

def TestSpecTypeClassifier(text):

    model = pickle.load(open(typeclassifier_model_path, 'rb'))
    TestSpec_Type = model.predict(text)
    
    return TestSpec_Type

if __name__ == "__main__":
    
    testspec_step = "Set  ' Calibration to set required coolant flow '  to 10litres/min     // INCA"
    
    print(f'Input Test Step:{testspec_step}\n')

    testspec_step, Var_Details, port = TestSpec_Preprocessing(testspec_step)
    
    print('Variable Identified :', Var_Details)
    print('Port :', port)

    TestSpec_Type = TestSpecTypeClassifier([testspec_step])
    print(f'TestSpec Type prediction : {TestSpec_Type}')

    text_token, label_pred = LHS_RHS_Prediction(testspec_step)
    
    for i in range(len(text_token)):
        print(f'Word : {text_token[i]} , LHS RHS Pred: {label_pred[i]}\n')


