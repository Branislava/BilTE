from nltk.tag import pos_tag, map_tag
from nltk.corpus import stopwords

class MWE:
    
    def __init__(self, name, mwe):
        
        # već _A_potencijalan _N_korisnik 
        # => [('već', '-'), ('potencijalan', 'A'), ('korisnik', 'N')]
        self.tagged_tokens = list()
        for token in mwe.split(' '):
            pera = token.split('_')
            
            # ['već']
            if len(pera) == 1:
                self.tagged_tokens.append((pera[0], 'INT'))
            # ['', 'A', 'potencijalan']
            else:
                for i in range(1, len(pera) - 1, 2):
                    self.tagged_tokens.append((pera[i+1], pera[i]))
        
        self.mwe = ' '.join(token for token, tag in self.tagged_tokens)
        self.name = name
        
        self.nr_of_tokens = len(self.tagged_tokens)
        self.nr_of_chars = len(self.mwe)
        
        self.stoplist = None
    
    def number_of_characters(self):
        return self.nr_of_chars
    
    def number_of_different_characters(self):
        return len(set(self.mwe.lower()))

    def number_of_tokens(self):
        return self.nr_of_tokens
    
    def avg_token_length(self):
        return 1.0 * sum([len(token) for token, _ in self.tagged_tokens]) / self.nr_of_tokens
    
    # WARNING: ako lose radi, probaj sa sirim skupom tagova
    def perc_of_nouns(self):
        tags = ['NOUN', 'N']
        return 1.0 * sum(1 for _, tag in self.tagged_tokens if tag in tags) / self.nr_of_tokens
    
    def perc_of_verbs(self):
        tags = ['VERB', 'V']
        return 1.0 * sum(1 for _, tag in self.tagged_tokens if tag in tags) / self.nr_of_tokens
    
    def perc_of_adjectives(self):
        tags = ['ADJ', 'A']
        return 1.0 * sum(1 for _, tag in self.tagged_tokens if tag in tags) / self.nr_of_tokens
    
    def perc_of_numerals(self):
        tags = ['NUM']
        return 1.0 * sum(1 for _, tag in self.tagged_tokens if tag in tags) / self.nr_of_tokens
    
    def perc_of_pronouns(self):
        tags = ['PRO', 'PRON']
        return 1.0 * sum(1 for _, tag in self.tagged_tokens if tag in tags) / self.nr_of_tokens
    
    def perc_of_conjuctions(self):
        tags = ['CONJ', 'INT']
        return 1.0 * sum(1 for _, tag in self.tagged_tokens if tag in tags) / self.nr_of_tokens
    
    def perc_of_adverbs(self):
        tags = ['ADV']
        return 1.0 * sum(1 for _, tag in self.tagged_tokens if tag in tags) / self.nr_of_tokens
    
    def perc_of_prepositions(self):
        tags = ['PREP', 'ADP', 'PRT']
        return 1.0 * sum(1 for _, tag in self.tagged_tokens if tag in tags) / self.nr_of_tokens
    
    def perc_of_stopwords(self):
        return 1.0 * sum(1 for token, _ in self.tagged_tokens if token in self.stoplist) / self.nr_of_tokens
    
    def perc_vocab_richness(self):
        return 1.0 * len(set(self.tagged_tokens)) / self.nr_of_tokens

    def tag_0(self):
        return self.tagged_tokens[0][1]

    def tag_1(self):
        tag = '-'
        if self.nr_of_tokens > 1:
            tag = self.tagged_tokens[1][1]
        return tag

    def tag_2(self):
        tag = '-'
        if self.nr_of_tokens > 2:
            tag = self.tagged_tokens[2][1]
        return tag
    
    def tag_3(self):
        tag = '-'
        if self.nr_of_tokens > 3:
            tag = self.tagged_tokens[3][1]
        return tag
    
    def tag_4(self):
        tag = '-'
        if self.nr_of_tokens > 4:
            tag = self.tagged_tokens[4][1]
        return tag
    
    def tag_5(self):
        tag = '-'
        if self.nr_of_tokens > 5:
            tag = self.tagged_tokens[5][1]
        return tag

    def perc_of_consonants(self):
        return 1.0 * sum(1 for letter in self.mwe if letter.isalpha() and letter.lower() not in 'aeiou') / self.nr_of_chars

    def perc_of_vocals(self):
        return 1.0 * sum(1 for letter in self.mwe if letter.lower() in 'aeiou') / self.nr_of_chars

    def perc_lexical_diversity(self):
        return 1.0 * self.number_of_different_characters() / self.nr_of_chars
    
    def perc_of_diacritics(self):
        return 1.0 * sum(1 for letter in self.mwe if letter.lower() in 'ščćđž') / self.nr_of_chars
    
    def perc_tokens_longer_10(self):
        return 1.0 * sum(1 for token, _ in self.tagged_tokens if len(token) >= 10) / self.nr_of_tokens
    
    def perc_tokens_longer_8(self):
        return 1.0 * sum(1 for token, _ in self.tagged_tokens if len(token) >= 8) / self.nr_of_tokens
    
    def perc_tokens_longer_6(self):
        return 1.0 * sum(1 for token, _ in self.tagged_tokens if len(token) >= 6) / self.nr_of_tokens
    
    def perc_tokens_shorter_5(self):
        return 1.0 * sum(1 for token, _ in self.tagged_tokens if len(token) <= 5) / self.nr_of_tokens
    
    def perc_tokens_shorter_4(self):
        return 1.0 * sum(1 for token, _ in self.tagged_tokens if len(token) <= 4) / self.nr_of_tokens
    
    def perc_tokens_shorter_3(self):
        return 1.0 * sum(1 for token, _ in self.tagged_tokens if len(token) <= 3) / self.nr_of_tokens
    
    def is_compound(self):
        return any([True for token, _ in self.tagged_tokens if '-' in token])
        
class SerbianMWE(MWE):
    
    def __init__(self, name, mwe):
        
        super(SerbianMWE, self).__init__(name, mwe)
        self.stoplist = ['a','ako','ali','bi','bih','bila','bili','bilo','bio','bismo','biste','biti','bumo','da','do','duž','ga','hoće','hoćemo','hoćete','hoćeš','hoću','i','iako','ih','ili','iz','ja','je','jedna','jedne','jedno','jer','jesam','jesi','jesmo','jest','jeste','jesu','jim','joj','još','ju','kada','kako','kao','koja','koje','koji','kojima','koju','kroz','li','me','mene','meni','mi','mimo','moj','moja','moje','mu','na','nad','nakon','nam','nama','nas','naš','naša','naše','našeg','ne','nego','neka','neki','nekog','neku','nema','netko','neće','nećemo','nećete','nećeš','neću','nešto','ni','nije','nikoga','nikoje','nikoju','nisam','nisi','nismo','niste','nisu','njega','njegov','njegova','njegovo','njemu','njezin','njezina','njezino','njih','njihov','njihova','njihovo','njim','njima','njoj','nju','no','o','od','odmah','on','ona','oni','ono','ova','pa','pak','po','pod','pored','prije','s','sa','sam','samo','se','sebe','sebi','si','smo','ste','su','sve','svi','svog','svoj','svoja','svoje','svom','ta','tada','taj','tako','te','tebe','tebi','ti','to','toj','tome','tu','tvoj','tvoja','tvoje','u','uz','vam','vama','vas','vaš','vaša','vaše','već','vi','vrlo','za','zar','će','ćemo','ćete','ćeš','ću','što']

class EnglishMWE(MWE):  
    
    def __init__(self, name, mwe):
        
        super(EnglishMWE, self).__init__(name, mwe)
        self.stoplist = set(stopwords.words('english'))
