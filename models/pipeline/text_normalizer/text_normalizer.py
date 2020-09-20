import nltk
import re
import json

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

class TextNormalizer:    

    def lower_case(self, text_ser):
        text_ser = text_ser.str.lower()

        return text_ser

    def remove_extra_spaces(self, text_ser):
        text_ser = text_ser.str.replace(r'\n', ' ')
        text_ser = text_ser.str.replace(r'\s+', ' ')

        return text_ser

    def get_text_expansion_rule(self):
        with open('models/pipeline/text_normalizer/text_expansion_rule.json', 'r') as f:
            expansion_rule = json.loads(f.read())
        return expansion_rule

    def expand_abbreviation(self, text_ser):
        expansion_rule = self.get_text_expansion_rule()
        c_re = re.compile('(%s)' % '|'.join(expansion_rule.keys()))

        def expandContractions(text):
            def replace(match):
                return expansion_rule[match.group(0)]
            return c_re.sub(replace, text)

        text_ser =text_ser.apply(expandContractions)

        return text_ser      

    def remove_non_english(self, text_ser):
        text_ser = text_ser.str.replace(r'[^a-zA-Z]', ' ')

        return text_ser    

    def normalize_text(self, text_ser, stop_words_removal=True):
            
        lemmatizer = WordNetLemmatizer()

        normalized_text_list = []

        def get_JNVR(word_type):
            if word_type[0] in ["J", "N", "V", "R"]:
                return True
            else:
                return False
            
        def get_wordnet_pos(word_type):
            """Map POS tag to first character lemmatize() accepts"""
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}

            return tag_dict.get(word_type[0], wordnet.NOUN)
        
        if stop_words_removal == True:
            stop_words = set(stopwords.words('english')) 
            text_word_list = text_ser.apply(lambda text: [word for word in word_tokenize(text) if not word in stop_words])
        else:
            text_word_list = text_ser.apply(lambda text: [word for word in word_tokenize(text)])

        text_word_list = [nltk.pos_tag(word_list) for word_list in text_word_list]

        for word_list in text_word_list:            
            normalized_text_list.append(' '.join([lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1])) for word in word_list if get_JNVR(word[1])==True]))
                    
        return normalized_text_list