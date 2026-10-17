import spacy
from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin
import os
import pandas as pd  # Added missing import

nlp = spacy.load('en_core_web_sm')

def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        
        # Lemmatize nouns and verbs (except the word 'left')
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
            
        pos_list.append(token.pos_)
        
    return word_list, pos_list

def process_humanml3d(corpus):
    text_save_path = './data/texts' # Updated to match your path
    os.makedirs(text_save_path, exist_ok=True) # Prevents folder missing errors
    
    for i in tqdm(range(len(corpus)), desc="Processing HumanML3D"):
        caption = corpus.iloc[i]['caption']
        start = corpus.iloc[i]['from']
        end = corpus.iloc[i]['to']
        name = corpus.iloc[i]['new_joint_name']
        
        word_list, pos_list = process_text(caption)
        
        # Safely zip lists together to avoid index mismatches
        tokens = ' '.join(['%s/%s' % (w, p) for w, p in zip(word_list, pos_list)])
        
        file_path = pjoin(text_save_path, name.replace('npy', 'txt'))
        with cs.open(file_path, 'a+', encoding='utf-8') as f:
            f.write('%s#%s#%s#%s\n' % (caption, tokens, start, end))

def process_kitml(corpus):
    text_save_path = './data/texts' # Updated to match your path
    os.makedirs(text_save_path, exist_ok=True)
    
    for i in tqdm(range(len(corpus)), desc="Processing KIT-ML"):
        caption = corpus.iloc[i]['desc']
        start = 0.0
        end = 0.0
        name = corpus.iloc[i]['data_id']
        
        word_list, pos_list = process_text(caption)
        tokens = ' '.join(['%s/%s' % (w, p) for w, p in zip(word_list, pos_list)])
        
        file_path = pjoin(text_save_path, str(name) + '.txt')
        with cs.open(file_path, 'a+', encoding='utf-8') as f:
            f.write('%s#%s#%s#%s\n' % (caption, tokens, start, end))

if __name__ == "__main__":
    # Update this path if your CSV file is inside ./data/ instead of ./dataset/
    csv_path = './dataset/kit_mocap_dataset/desc_final.csv'
    
    if os.path.exists(csv_path):
        corpus = pd.read_csv(csv_path)
        process_humanml3d(corpus)
    else:
        print(f"Error: CSV file not found at {csv_path}")
