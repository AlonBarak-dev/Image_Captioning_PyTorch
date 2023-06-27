import spacy
from collections import Counter


class Vocabulary:
    """
        The main purpose of this class is to convert embedded outputs
        of the model to a readable sentence.
        It saves a "Bag of words" of all words in the dataset and allocate
        an unique index for each word.
        For words outside of the Bag of words, it will use the <UNK> token.
    """
    
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
        self.spacy_eng = spacy.load("en_core_web_sm")
        
    def __len__(self): 
        return len(self.itos)
    
    def tokenize(self, text):
        return [token.text.lower() for token in self.spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4 # 0-3 allready allocated
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ 
            For each word in the text corresponding index 
            token for that word form the vocab built as list
        """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]    