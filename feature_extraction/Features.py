class Features:
    
    def __init__(self):
        self.feature_map = None
    
    def extract(self):
        pass


class SingleFeatures(Features):
    
    def __init__(self, MWE):
    
        self.MWE = MWE
        
        self.feature_map = {
            'number_of_characters': self.MWE.number_of_characters,
            'number_of_different_characters': self.MWE.number_of_different_characters,
            'number_of_tokens': self.MWE.number_of_tokens,
            'avg_token_length': self.MWE.avg_token_length,
            'perc_of_nouns': self.MWE.perc_of_nouns,
            'perc_of_verbs': self.MWE.perc_of_verbs,
            'perc_of_adjectives': self.MWE.perc_of_adjectives,
            'perc_of_numerals': self.MWE.perc_of_numerals,
            'perc_of_pronouns': self.MWE.perc_of_pronouns,
            'perc_of_conjuctions': self.MWE.perc_of_conjuctions,
            'perc_of_adverbs': self.MWE.perc_of_adverbs,
            'perc_of_prepositions': self.MWE.perc_of_prepositions,
            'perc_of_stopwords': self.MWE.perc_of_stopwords,
            'perc_vocab_richness': self.MWE.perc_vocab_richness,
            'tag_0': self.MWE.tag_0,
            'tag_1': self.MWE.tag_1,
            'tag_2': self.MWE.tag_2,
            'tag_3': self.MWE.tag_3,
            'tag_4': self.MWE.tag_4,
            'tag_5': self.MWE.tag_5,
            'perc_of_consonants': self.MWE.perc_of_consonants,
            'perc_of_vocals': self.MWE.perc_of_vocals,
            'perc_of_diacritics': self.MWE.perc_of_diacritics,
            'perc_lexical_diversity': self.MWE.perc_lexical_diversity,
            'perc_tokens_longer_10': self.MWE.perc_tokens_longer_10,
            'perc_tokens_longer_8': self.MWE.perc_tokens_longer_8,
            'perc_tokens_longer_6': self.MWE.perc_tokens_longer_6,
            'perc_tokens_shorter_5': self.MWE.perc_tokens_shorter_5,
            'perc_tokens_shorter_4': self.MWE.perc_tokens_shorter_4,
            'perc_tokens_shorter_3': self.MWE.perc_tokens_shorter_3,
            
            'is_compound': self.MWE.is_compound,
        }
        
    def extract(self):
        
        feat_dict = dict()
        for feature in self.feature_map:
            feat_dict['%s_%s' % (self.MWE.name, feature)] = self.feature_map[feature]()
        return feat_dict
        
class JointFeatures(Features):
    
    def __init__(self, MWEs):
        
        self.MWEs = MWEs
        self.pairs = [(self.MWEs[i], self.MWEs[j]) for i in range(len(self.MWEs)) for j in range(i+1, len(self.MWEs))]
        
        self.feature_mappings = {
            'common_first_letters_1': self.common_first_letters_1,
            'common_first_letters_2': self.common_first_letters_2,
            'common_first_letters_3': self.common_first_letters_3,
            'common_substring_2': self.common_substring_2,
            'common_substring_3': self.common_substring_3,
            'common_substring_4': self.common_substring_4,
            'common_substring_longer_5': self.common_substring_longer_5,
            'common_substring_longer_6': self.common_substring_longer_6,
            'perc_of_common_tokens': self.perc_of_common_tokens
        }
        
    def extract(self):
        feat_dict = dict()
        for mapping in self.feature_mappings:
            for mwe1, mwe2 in self.pairs:
                feat_dict['{0}_{1}_{2}'.format(mwe1.name, mwe2.name, mapping)] = self.feature_mappings[mapping](mwe1, mwe2)
        return feat_dict
        
    def common_first_letters_1(self, mwe1, mwe2):
        n = 1
        return any(True for tok1, _ in mwe1.tagged_tokens for tok2, _ in mwe2.tagged_tokens if tok1.lower()[:n] == tok2.lower()[:n])
    
    def common_first_letters_2(self, mwe1, mwe2):
        n = 2
        return any(True for tok1, _ in mwe1.tagged_tokens for tok2, _ in mwe2.tagged_tokens if tok1.lower()[:n] == tok2.lower()[:n])
    
    def common_first_letters_3(self, mwe1, mwe2):
        n = 3
        return any(True for tok1, _ in mwe1.tagged_tokens for tok2, _ in mwe2.tagged_tokens if tok1.lower()[:n] == tok2.lower()[:n])
    
    # stackoverflow
    def substringFinder(self, string1, string2):
        answer = ""
        anslist=[]
        len1, len2 = len(string1), len(string2)
        for i in range(len1):
            match = ""
            for j in range(len2):
                if (i + j < len1 and string1[i + j] == string2[j]):
                    match += string2[j]
                else:
                    #if (len(match) > len(answer)): 
                    answer = match
                    if answer != '' and len(answer) > 1:
                        anslist.append(answer)
                    match = ""

            if match != '':
                anslist.append(match)
            # break
        return anslist
    
    def common_substring_2(self, mwe1, mwe2):
        n = 2
        substrings = self.substringFinder(mwe1.mwe, mwe2.mwe)
        return any(True for substr in substrings if len(substr) == n)
    
    def common_substring_3(self, mwe1, mwe2):
        n = 3
        substrings = self.substringFinder(mwe1.mwe, mwe2.mwe)
        return any(True for substr in substrings if len(substr) == n)
    
    def common_substring_4(self, mwe1, mwe2):
        n = 4
        substrings = self.substringFinder(mwe1.mwe, mwe2.mwe)
        return any(True for substr in substrings if len(substr) == n)
    
    def common_substring_longer_5(self, mwe1, mwe2):
        n = 5
        substrings = self.substringFinder(mwe1.mwe, mwe2.mwe)
        return any(True for substr in substrings if len(substr) >= n)
    
    def common_substring_longer_6(self, mwe1, mwe2):
        n = 6
        substrings = self.substringFinder(mwe1.mwe, mwe2.mwe)
        return any(True for substr in substrings if len(substr) >= n)
    
    def perc_of_common_tokens(self, mwe1, mwe2):
        common_tokens = set([token for token, _ in mwe1.tagged_tokens]).intersection(set([token for token, _ in mwe2.tagged_tokens]))
        return 1.0 * 2 * len(common_tokens) / (mwe1.nr_of_tokens + mwe2.nr_of_tokens)