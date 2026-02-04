import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token ]


    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.vocab = []
        for element in self.special_tokens:
            self.vocab.append(element) 
        
        master_text = [word for text in texts for word in text.split()]
        
        for word in master_text: 
            if word not in self.vocab: 
                self.vocab.append(word) 

    

            


        for pair in enumerate(self.vocab): 
            self.word_to_id[pair[1]] =pair[0] 
            self.id_to_word[pair[0]] = pair[1] 
       
        self.vocab_size = len(self.vocab) 
    
    

    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """ 

        master_text = [word for word in text.split()]
        input_ids = []

        #input_ids.append(self.word_to_id[self.bos_token])
        
        for word in master_text: 
            if word in self.vocab: 
                input_ids.append(self.word_to_id[word])
            else:
                input_ids.append(self.word_to_id[self.unk_token])

        #input_ids.append(self.word_to_id[self.eos_token])
       
        return input_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        back_to_text = []
        
        for input_id in ids: 
            if self.id_to_word[input_id] in self.special_tokens: 
                continue
            else:
                back_to_text.append(self.id_to_word[input_id])
            

        return " ".join(back_to_text) 





