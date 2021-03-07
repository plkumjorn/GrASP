from typing import Iterable, List, Set, Callable, Optional, Union, Sequence
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
from termcolor import colored
import numpy as np
import csv
import json
import math
import random
import nltk
import spacy
nlp = spacy.load('en_core_web_sm')
tokenizer = spacy.load('en_core_web_sm', disable = ['tagger', 'parser', 'ner', 'textcat']) # Use only tokenizer

# ========== Utils ==========
def entropy_binary(count_pos: int, count_neg: int) -> float:
    n_total = count_pos + count_neg
    if n_total == 0:
        return None
    p_pos = count_pos / n_total
    p_neg = count_neg / n_total
    if p_pos == 0 or p_neg == 0:
        return 0.0
    else:
        return -(p_pos * math.log2(p_pos) + p_neg * math.log2(p_neg))

# ========== Attributes ==========
def default_attr2text(attr:str, 
                      is_complement:bool = False) -> str:
    if is_complement:
        return f'having an attribute {attr}'
    else:
        return f'a word with an attribute {attr}'

class Attribute():
    
    def __init__(self, 
                 name: str,
                 extraction_function: Callable[[str, List[str]], List[Set[str]]],
                 translation_function: Callable[[str, bool], str] = default_attr2text,
                 values: Optional[Iterable[str]] = None # Unique binary values of this attribute
                ) -> None:
        self.name = name
        self.extraction_function = extraction_function
        self.translation_function = translation_function
        self.values = values
        
    def __str__(self) -> str:
        if self.values is not None:
            ans = f'{self.name}: {self.values}'
            return ans
        return self.name
    
    def extract(self, text: str, tokens: List[str]) -> List[Set[str]]:
        pre_ans = self.extraction_function(text, tokens)
        ans = [set([f'{self.name}:{item}' for item in t_pre_ans]) for t_pre_ans in pre_ans]
        return ans
        
        
class CustomAttribute(Attribute):
    
    def __init__(self, 
                 name: str,
                 extraction_function: Callable[[str, List[str]], List[Set[str]]],
                 translation_function: Callable[[str, bool], str] = default_attr2text,
                 values: Optional[Iterable[str]] = None # Unique binary values of this attribute
                ) -> None:
        super().__init__(name, extraction_function, translation_function, values)

# ----- Text attribute -----        
def _text_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
    tokens = map(str.lower, tokens)
    return [set([t]) for t in tokens]

def _text_translation(attr:str, 
                      is_complement:bool = False) -> str:
    word = attr[5:]
    return f'the word "{word}"'

TextAttribute = Attribute(name = 'TEXT', extraction_function = _text_extraction, translation_function = _text_translation)

# ----- Spacy attribute (POS, DEP, NER) -----
def _spacy_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
    ans = []
    for t in nlp(text):
        t_ans = []
        t_ans.append(f'POS-{t.pos_}') # Universal dependency tag https://universaldependencies.org/u/pos/
#         t_ans.append(f'POS-{t.tag_}') # Penn tree bank Part-of-speech https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        t_ans.append(f'DEP-{t.dep_}') # Dependency parsing tag
        if t.ent_type_ != "":
            t_ans.append(f'NER-{t.ent_type_}') # Named-entity recognition
        ans.append(set(t_ans))
    return ans

def _spacy_translation(attr:str, 
                      is_complement:bool = False) -> str:
    subtype = attr[6:9]
    value = attr[10:]

    if subtype == 'POS':
        mapping = {
            'ADJ': 'an adjective (describes a noun)',
            'ADP': 'a perposition',
            'ADV': 'an adverb (describes a verb)',
            'AUX': 'an auxiliary (a function word for verbs)',
            'CCONJ': 'a coordinating conjunction (links parts of the sentence)',
            'DET': 'a determiner (expresses a reference of a noun)',
            'INTJ': 'an interjection (expresses an emotional reaction)',
            'NOUN': 'a noun',
            'NUM': 'a number',
            'PART': 'a function word of type particle',
            'PRON': 'a pronoun',
            'PROPN': 'a proper noun (a name of a specific individual, place, or object)',
            'PUNCT': 'a punctuation mark',
            'SCONJ': 'a subordinating conjunction (defines a relation between sentence parts)',
            'SYM': 'a symbol',
            'VERB': 'a verb',
            'X': 'a word of category misc.',
            'SPACE': 'a space',
            'EOL': 'an end of line (EOL)'
        }
        return mapping[value]

    elif subtype == 'DEP':
        if is_complement:
            return f'having the dependency type "{value}"'
        else:
            return f'a word with the dependency type "{value}"'

    elif subtype == 'NER':
        mapping = {
            "PERSON": "a named person",
            "NORP": "a nationality or a religious or political group",
            "FACILITY": "a facility (such as a building, an airport, etc.)",
            "FAC": "a facility (such as a building, an airport, etc.)",
            "ORG": "an organization (such as a company, an agency, etc.)",
            "GPE": "a country, city, or state",
            "LOC": "a non-geopolitical location (such as a mountain range, a body of water)",
            "PRODUCT": "a named product",
            "EVENT": "a named event (such as a hurricane, a battle, etc.)",
            "WORK_OF_ART": "a title of a work of art (such as a book, a song, etc.)",
            "LAW": "a named document made into laws",
            "LANGUAGE": "a named language",
            "DATE": "an absolute or relative date or period",
            "TIME": "a time smaller than a day",
            "PERCENT": 'a percentage (including "%")',
            "MONEY": "a monetary value",
            "QUANTITY": "a measurement",
            "ORDINAL": 'an ordinal number (such as "first", "second", etc.)',
            "CARDINAL": 'a cardinal number (such as one, two, etc.)',
        }
        return mapping[value]

    else:
        assert False, f"Invalid spacy attribute {attr}"

SpacyAttribute = Attribute(name = 'SPACY', extraction_function = _spacy_extraction, translation_function = _spacy_translation)

# ----- Hypernym attribute -----
HYPERNYM_DICT = dict()

def _get_anvr_pos(penn_tag: str): # Return Literal['a', 'n', 'v', 'r', None]
    if penn_tag.startswith('JJ'): # Adjective
        return 'a'
    elif penn_tag.startswith('NN'): # Noun
        return 'n'
    elif penn_tag.startswith('VB'): # Verb
        return 'v'
    elif penn_tag.startswith('RB'): # Adverb
        return 'r'
    else: # Invalid wordnet type
        return None 
    
def _get_all_hypernyms(synset: nltk.corpus.reader.wordnet.Synset) -> Set[nltk.corpus.reader.wordnet.Synset]:
    if str(synset) not in HYPERNYM_DICT:
        ans = set()
        direct_hypernyms = synset.hypernyms() # type: List[nltk.corpus.reader.wordnet.Synset]
        for ss in direct_hypernyms:
            ans.update(_get_all_hypernyms(ss))
            ans.update(set([ss]))
        HYPERNYM_DICT[str(synset)] = ans
    return HYPERNYM_DICT[str(synset)]
    
def _hypernym_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
    ans = []
    for t in nlp(text):
        pos = _get_anvr_pos(t.tag_)
        if pos is not None:
            synset = lesk(tokens, t.text, pos)
            if synset is not None:
                all_hypernyms = set([synset]) # This version of hypernym extraction includes synset of the word itself
                all_hypernyms.update(_get_all_hypernyms(synset))
                ans.append(set([str(ss)[8:-2] for ss in all_hypernyms]))
            else:
                ans.append(set([]))
        else:
            ans.append(set([]))
    return ans

def _hypernym_translation(attr:str, 
                      is_complement:bool = False) -> str:
    synset = attr.split(":")[1].split('.')
    word, pos, idx = synset[0], synset[1], synset[2]
    if pos == 'a': pos = 'adj'
    if pos == 'r': pos = 'adv' 
    return f'a type of {word} ({pos})'

HypernymAttribute = Attribute(name = 'HYPERNYM', extraction_function = _hypernym_extraction, translation_function = _hypernym_translation)
        
# ----- Sentiment attribute -----        
# Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In International Conference on Knowledge Discovery and Data Mining, KDD’04, pages 168–177. (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)

POSITIVE_LEXICON = [line.strip().lower() for line in open('resources/opinion-lexicon-English/positive-words.txt') if line.strip() != '' and line[0] != ';']
NEGATIVE_LEXICON = [line.strip().lower() for line in open('resources/opinion-lexicon-English/negative-words.txt') if line.strip() != '' and line[0] != ';']

def _sentiment_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
    tokens = map(str.lower, tokens)
    ans = []
    for t in tokens:
        t_ans = []
        if t.lower() in POSITIVE_LEXICON:
            t_ans.append('pos')
        if t.lower() in NEGATIVE_LEXICON:
            t_ans.append('neg')
        ans.append(set(t_ans))
    return ans

def _sentiment_translation(attr:str, 
                      is_complement:bool = False) -> str:
    sentiment = attr.split(":")[1]
    sentiment = 'positive' if sentiment == 'pos' else 'negative'
    if is_complement:
        return f'bearing a {sentiment} sentiment'
    else:
        return f'a {sentiment}-sentiment word'
    
SentimentAttribute = Attribute(name = 'SENTIMENT', extraction_function = _sentiment_extraction, translation_function = _sentiment_translation, values = ['pos', 'neg'])

# ALL_ATTRIBUTES = {attr.name:attr for attr in [TextAttribute, SpacyAttribute, HypernymAttribute, SentimentAttribute]}
# ========== Augmented Text ==========

class AugmentedText():
    
    def __init__(self, 
                 text: str,
                 is_positive: bool = False, 
                 include_standard: List[str] = ['TEXT', 'POS', 'DEP', 'NER', 'HYPERNYM', 'SENTIMENT'], 
                 include_custom: List[CustomAttribute] = []) -> None:
        self.text = text
        self.include_standard = include_standard
        self.include_custom = include_custom
        self.tokens = [t.text for t in tokenizer(self.text)]
        self.features = dict()
        self.attributes = []
        
        # Standard attributes
        if 'TEXT' in self.include_standard:
            self.attributes.append(TextAttribute)
        if set(include_standard).intersection(set(['POS', 'DEP', 'NER'])):
            self.attributes.append(SpacyAttribute)
        if 'HYPERNYM' in self.include_standard:
            self.attributes.append(HypernymAttribute)
        if 'SENTIMENT' in self.include_standard:
            self.attributes.append(SentimentAttribute)
            
        # Custom attributes
        self.attributes = self.attributes + self.include_custom
        attribute_names = [attr.name for attr in self.attributes]
        assert len(attribute_names) == len(set(attribute_names)), "Attribute names are not unique. Please do not use TEXT, POS, DEP, HYPERNYM, NER, SENTIMENT as custom attribute names"
        
        # Extraction
        for attr in self.attributes:
            # Filter out 'POS', 'DEP', 'NER' if they are excluded
            if attr.name == 'SPACY':
                ans = attr.extract(self.text, self.tokens)
                for i in range(len(ans)):
                    filtered_ans = []
                    for attr_type in ['POS', 'DEP', 'NER']:
                        if attr_type in self.include_standard:
                            filtered_ans += [item for item in ans[i] if item.startswith(f'SPACY:{attr_type}')]
                    ans[i] = set(filtered_ans)
                self.features[attr.name] = ans
            else:
                self.features[attr.name] = attr.extract(self.text, self.tokens)
            assert len(self.features[attr.name]) == len(self.tokens), f"The number of tokens returned by the extraction function of {attr.name} is not equal to tokens from Spacy."
            
        # Merge features from all attributes
        self.all_features = []
        for i in range(len(self.tokens)):
            t_ans = set()
            for attr_name, f in self.features.items():
                t_ans.update(f[i])
            self.all_features.append(t_ans)
        assert len(self.all_features) == len(self.tokens)
        
        # A set of all unique features in this text
        self.all_unique_features = set([item for t_ans in self.all_features for item in t_ans])
        
    def __str__(self) -> str:
        lines = [f'{t}: {f}' for t, f in zip(self.tokens, self.all_features)]
        return self.text + '\n' + '\n'.join(lines)
    
    def keep_only_features(self, features_to_keep: Iterable[str]) -> None:
        for i in range(len(self.all_features)):
            self.all_features[i] = self.all_features[i].intersection(features_to_keep)

# ========== Patterns ==========

class Pattern():
    
    def __init__(self,
                 pattern: List[Set[str]],
                 window_size: Optional[int], # None means no window size (can match an arbitrary length)
                 parent: Optional['Pattern'] = None,
                 grasp: Optional['GrASP'] = None,
                ) -> None:
        
        self.pattern = pattern
        self.parent = parent
        # self.grasp = grasp
        self.window_size = window_size
        self.print_examples = grasp.print_examples
        self.pos_augmented = grasp.pos_augmented
        self.neg_augmented = grasp.neg_augmented
        self.sort_key = grasp.sort_key
        self.gain_criteria = grasp.gain_criteria
        self.all_attributes_dict = grasp.all_attributes_dict
        
        # ----- Check pattern matching
        if self.pattern == []: # Root node matches everything
            assert self.parent is None, "Root node cannot have any parent"
            self.pos_example_labels = [True for item in self.pos_augmented]
            self.neg_example_labels = [True for item in self.neg_augmented]
             # True (Match parent and here) / False (Match parent but not here) / None (Not match in parent)
        else:
            if self.parent is None:
                self.pos_example_labels = [self.is_match(augtext)[0] for augtext in self.pos_augmented]
                self.neg_example_labels = [self.is_match(augtext)[0] for augtext in self.neg_augmented]
            else:
                self.pos_example_labels = [self.is_match(augtext)[0] if val else None for augtext, val in zip(self.pos_augmented, self.parent.pos_example_labels)]
                self.neg_example_labels = [self.is_match(augtext)[0] if val else None for augtext, val in zip(self.neg_augmented, self.parent.neg_example_labels)]
        
        # ----- Count match and notmatch
        pos_match, neg_match = self.pos_example_labels.count(True), self.neg_example_labels.count(True)
        pos_notmatch, neg_notmatch = self.pos_example_labels.count(False), self.neg_example_labels.count(False)
        pos_none, neg_none = self.pos_example_labels.count(None), self.neg_example_labels.count(None)
        
        self.num_total_match = pos_match + neg_match
        self.num_total_notmatch = pos_notmatch + neg_notmatch
        self.num_total_all = self.num_total_match + self.num_total_notmatch
        if self.parent is not None:
            assert self.num_total_all == self.parent.num_total_match
        self.prob_match = self.num_total_match / self.num_total_all
        self.prob_notmatch = self.num_total_notmatch / self.num_total_all
           
        # ----- Calculate entropy and information gain
        self.entropy_match = entropy_binary(pos_match, neg_match)
        self.entropy_notmatch = entropy_binary(pos_notmatch, neg_notmatch)
        if self.parent is None:
            self.information_gain = None
            self.relative_information_gain = None
        else:
            # print(self.parent.entropy_match, self.prob_match, self.entropy_match, self.prob_notmatch, self.entropy_notmatch)
            self.information_gain = self.parent.entropy_match
            if self.prob_match > 0:
                self.information_gain -= self.prob_match * self.entropy_match
            if self.prob_notmatch > 0:
                self.information_gain -= self.prob_notmatch * self.entropy_notmatch
            self.relative_information_gain = self.information_gain / self.parent.entropy_match if self.parent.entropy_match != 0 else 0
                
        # ----- Calculate global weighted entropy and information gain (Consider None as False)
        self.root_entropy = entropy_binary(len(self.pos_augmented), len(self.neg_augmented))
        self.global_entropy_match = self.entropy_match
        self.global_entropy_notmatch = entropy_binary(pos_notmatch+pos_none, neg_notmatch+neg_none)
        self.global_num_total_all = len(self.pos_augmented) + len(self.neg_augmented)
        self.global_prob_match = self.num_total_match / self.global_num_total_all
        self.global_prob_notmatch = 1 - self.global_prob_match
        if self.parent is None:
            self.global_information_gain = None
        else:
            self.global_information_gain = self.root_entropy
            if self.global_prob_match > 0:
                self.global_information_gain -= self.global_prob_match * self.global_entropy_match
            if self.global_prob_notmatch > 0:
                self.global_information_gain -= self.global_prob_notmatch * self.global_entropy_notmatch
        
        # ----- Calculate precision, recall, and coverage of the pattern
        self.support_class = "Positive" if pos_match >= neg_match else "Negative"
        self.precision = max(pos_match, neg_match) / self.num_total_match if self.num_total_match > 0 else None
        self.recall = pos_match / len(self.pos_example_labels) if pos_match >= neg_match else neg_match / len(self.neg_example_labels)
        self.coverage = self.global_prob_match
        try:
            self.metric = self.sort_key(self)
        except:
            print(self.precision, self.recall)
            raise Exception()
    
    def normalized_mutual_information(self, another_pattern: 'Pattern') -> float:
        labels_1 = self.get_all_labels()
        labels_2 = another_pattern.get_all_labels()
        assert len(labels_1) == len(labels_2)
        assert self.parent == another_pattern.parent
        
        f_labels_1, f_labels_2 = [], []
        for l1, l2 in zip(labels_1, labels_2):
            assert (l1 is None) == (l2 is None)
            if l1 is not None:
                f_labels_1.append(int(l1))
                f_labels_2.append(int(l2))
        return normalized_mutual_info_score(f_labels_1, f_labels_2)
    
    def global_normalized_mutual_information(self, another_pattern: 'Pattern') -> float:
        labels_1 = self.global_get_all_labels()
        labels_2 = another_pattern.global_get_all_labels()
        assert len(labels_1) == len(labels_2)
        return normalized_mutual_info_score(labels_1, labels_2)
        
    @staticmethod
    def _is_token_match(pattern_attributes: Set[str], token_attributes: Set[str]) -> bool:
        return pattern_attributes.issubset(token_attributes)
    
        
    def _is_match_recursive(self, pattern: List[Set[str]], attribute_list: List[Set[str]], start: int, end: Optional[int]) -> List[int]:
        # Base case
        if pattern == []:
            return []
        if start == len(attribute_list):
            return False
        
        # Recursive case
        if end is None:
            stop_match = len(attribute_list)
        else:
            assert end <= len(attribute_list)
            stop_match = end
        for idx in range(start, stop_match):
            if Pattern._is_token_match(pattern[0], attribute_list[idx]):
                if self.window_size is None: # No window size
                    match_indices = self._is_match_recursive(pattern[1:], attribute_list, idx+1, len(attribute_list))
                else: # Has window size
                    if end is None: # Match the first token (The end point has not been fixed)
                        match_indices = self._is_match_recursive(pattern[1:], attribute_list, idx+1, min(idx+self.window_size, len(attribute_list)))
                    else: # The end point has been fixed
                        match_indices = self._is_match_recursive(pattern[1:], attribute_list, idx+1, end)
                if isinstance(match_indices, list):
                    return [idx] + match_indices
        return False
                            
    def is_match(self, augtext: AugmentedText) -> [bool, Union[List[int], bool]]:
        match_indices = self._is_match_recursive(self.pattern, augtext.all_features, 0, None)
        return isinstance(match_indices, list), match_indices        
    
    def get_all_labels(self) -> List[Optional[bool]]:
        return self.pos_example_labels + self.neg_example_labels
    
    def global_get_all_labels(self) -> List[bool]:
        ans = self.pos_example_labels + self.neg_example_labels
        ans = [item if item is not None else False for item in ans] # Change None to False
        return ans
    
    @staticmethod
    def pattern_list2str(pattern: List[Set[str]]) -> str:
        ans = [sorted(list(a_set)) for a_set in pattern]
        return str(ans)
    
    def get_pattern_id(self) -> str:
        return Pattern.pattern_list2str(self.pattern)
    
    def _print_match_text(self, augtext: AugmentedText) -> str:
        is_match, match_indices = self.is_match(augtext)
        if not is_match:
            return colored('[NOT MATCH]', 'red', attrs=['blink']) + ': ' + ' '.join(augtext.tokens)
        else:
            ans = colored('[MATCH]', 'green', attrs=['blink']) + ': '
            assert isinstance(match_indices, list)
            for idx, t in enumerate(augtext.tokens):
                if idx not in match_indices:
                    ans += t + ' '
                else:
                    ans += colored(f'{t}:{list(self.pattern[match_indices.index(idx)])}', 'cyan', attrs=['reverse']) + ' '
            return ans
    def get_example_string(self, num_examples: int = 2) -> str:
        num_print_ex = num_examples
        example_string = ''
        if num_print_ex > 0:
            example_string = '\n' + colored('Examples', 'green', attrs=['reverse', 'blink']) + f' ~ Class {self.support_class}:\n'
            examples = []
            if self.support_class == 'Positive':
                ids = [idx for idx, v in enumerate(self.pos_example_labels) if v]
                random.shuffle(ids)
                for id in ids[:num_print_ex]:
                    examples.append(self._print_match_text(self.pos_augmented[id]))
                    examples.append('-'*25)
            else:
                ids = [idx for idx, v in enumerate(self.neg_example_labels) if v]
                random.shuffle(ids)
                for id in ids[:num_print_ex]:
                    examples.append(self._print_match_text(self.neg_augmented[id]))
                    examples.append('-'*25)
            example_string += '\n'.join(examples)
        return example_string
    
    def get_counterexample_string(self, num_examples: int = 2) -> str:
        num_print_counterex = num_examples
        counterexample_string = ''
        if num_print_counterex > 0:
            counterexample_string = '\n' + colored('Counterexamples', 'red', attrs=['reverse', 'blink']) + f' ~ Not class {self.support_class}:\n'
            counterexamples = []
            if self.support_class == 'Positive':
                ids = [idx for idx, v in enumerate(self.neg_example_labels) if v]
                random.shuffle(ids)
                for id in ids[:num_print_counterex]:
                    counterexamples.append(self._print_match_text(self.neg_augmented[id]))
                    counterexamples.append('-'*25)
            else:
                ids = [idx for idx, v in enumerate(self.pos_example_labels) if v]
                random.shuffle(ids)
                for id in ids[:num_print_counterex]:
                    counterexamples.append(self._print_match_text(self.pos_augmented[id]))
                    counterexamples.append('-'*25)
            counterexample_string += '\n'.join(counterexamples)
        return counterexample_string
    
    def print_examples(self, num_examples: int = 2) -> None:
        print(self.get_example_string(num_examples))
        
    def print_counterexamples(self, num_examples: int = 2) -> None:
        print(self.get_counterexample_string(num_examples))
    
    def __str__(self) -> str:                   
        ans_list = [f'Pattern: {self.get_pattern_id()}',
                    f'Window size: {self.window_size}',
                    f'Class: {self.support_class}',
                    f'Precision: {self.precision:.3f}',
                    f'Match: {self.num_total_match} ({self.coverage*100:.1f}%)',
                    f'Gain = {self.global_information_gain:.3f}',
                   ]
        metric_name = '' if callable(self.gain_criteria) else f'({self.gain_criteria}) '
        ans_list.append(f'Metric {metric_name}= {self.metric:.3f}')
        example_string = self.get_example_string(num_examples = self.print_examples[0])
        counterexample_string = self.get_counterexample_string(num_examples = self.print_examples[1])
        return '\n'.join(ans_list) + example_string + counterexample_string + '\n' + ('='*50)
                 
            
# ========== GrASP ==========

class GrASP():
    
    def __init__(self,
                 min_freq_threshold: float = 0.005, # t1
                 correlation_threshold: float = 0.5, # t2
                 alphabet_size: int = 100, # k1
                 num_patterns: int = 100, # k2
                 max_len: int = 5, # maxLen
                 window_size: Optional[int] = 10, # w
                 gaps_allowed: Optional[int] = None, # If gaps allowed is not None, it overrules the window size
                 gain_criteria: Union[str, Callable[[Pattern], float]] = 'global', # 'F_beta', 'global', 'local', or 'relative'
                 min_coverage_threshold: Optional[float] = None, # float: Proportion of examples
                 print_examples: Union[int, Sequence[int]] = 2, 
                 include_standard: List[str] = ['TEXT', 'POS', 'DEP', 'NER', 'HYPERNYM', 'SENTIMENT'], 
                 include_custom: List[CustomAttribute] = []) -> None:
        
        # Standard hyperparameters: minimal frequency threshold t1=0.005, correlation threshold t2 =0.5, size of the alphabet k1 =100, number of patterns in the output k2 = 100, maximal pattern length maxLen=5, and window size w=10
        
        self.min_freq_threshold = min_freq_threshold
        self.correlation_threshold = correlation_threshold
        self.alphabet_size = alphabet_size
        self.num_patterns = num_patterns
        self.max_len = max_len
        
        # Gaps VS Window size
        if gaps_allowed is not None and gaps_allowed >= 0:
            self.gaps_allowed = gaps_allowed
            self.window_size = None
        elif gaps_allowed is not None and gaps_allowed < 0:
            raise Exception(f'Gaps allowed should not be less than 0, {gaps_allowed} given')
        else:
            self.gaps_allowed = None
            self.window_size = window_size
        
        # Gain criteria
        assert callable(gain_criteria) or (gain_criteria in ['global', 'local', 'relative']) or (gain_criteria.startswith('F_')), f"Gain criterial must be callable or 'F_beta' (beta is a float number), 'global', 'local', or 'relative', but {gain_criteria} is given"
        self.gain_criteria = gain_criteria
        
        if callable(self.gain_criteria):
            self.sort_key = self.gain_criteria
        elif self.gain_criteria == 'global':
            self.sort_key = lambda x: x.global_information_gain 
        elif self.gain_criteria == 'local':
            self.sort_key = lambda x: x.information_gain 
        elif self.gain_criteria == 'relative':
            self.sort_key = lambda x: x.relative_information_gain
        elif self.gain_criteria.startswith('F_'):
            try:
                beta = float(self.gain_criteria[2:])
            except:
                assert False, f"Invalid gain criteria: {self.gain_criteria}"
            self.sort_key = lambda x: (1+beta**2) * (x.precision * x.recall) / ((x.precision*beta**2) + x.recall) if (x.precision is not None) else 0.0
        else:
            assert False, f"Invalid gain criteria {self.gain_criteria}"

        # Minimum coverage
        self.min_coverage_threshold = min_coverage_threshold
        
        self.include_standard = include_standard
        self.include_custom = include_custom

        self.all_attributes_dict = dict()
        # Standard attributes
        if 'TEXT' in self.include_standard:
            self.all_attributes_dict['TEXT'] = TextAttribute
        if set(include_standard).intersection(set(['POS', 'DEP', 'NER'])):
            self.all_attributes_dict['SPACY'] = SpacyAttribute
        if 'HYPERNYM' in self.include_standard:
            self.all_attributes_dict['HYPERNYM'] = HypernymAttribute
        if 'SENTIMENT' in self.include_standard:
            self.all_attributes_dict['SENTIMENT'] = SentimentAttribute
        for attr_class in self.include_custom:
            self.all_attributes_dict[attr_class.name] = attr_class
        
        # For printing patterns
        if isinstance(print_examples, list) or isinstance(print_examples, tuple):
            assert len(print_examples) == 2 and all(i>=0 for i in print_examples)
            self.print_examples = print_examples
        else:
            assert print_examples >= 0
            self.print_examples = (print_examples, print_examples)
            
        self.candidate_alphabet = None # After removing non-frequent attributes
        self.alphabet = None
        self.positives = None
        self.negatives = None
    
    def _remove_nonfrequent_attributes(self) -> List[str]:
        assert self.pos_augmented is not None
        assert self.neg_augmented is not None
        
        all_augmented = self.pos_augmented + self.neg_augmented
        all_attributes = []
        for augtext in all_augmented:
            all_attributes += list(augtext.all_unique_features)
        the_counter = Counter(all_attributes)
        min_freq = self.min_freq_threshold * len(all_augmented)
        candidate_alphabet = []
        for the_attr, freq in the_counter.most_common():
            if freq < min_freq:
                break
            else:
                candidate_alphabet.append(the_attr)
            
        # Remove non-frequent attributes from all_features of augmented texts
        for augtext in self.pos_augmented:
            augtext.keep_only_features(candidate_alphabet)
        for augtext in self.neg_augmented:
            augtext.keep_only_features(candidate_alphabet)
            
        return candidate_alphabet
    
    def _find_top_k_patterns(self, patterns: List[Pattern], k: int, use_coverage_threshold: bool = True) -> List[Pattern]:
        patterns.sort(key = lambda x: x.metric, reverse = True) # Sort by information gain descendingly
        
        ans = []
        for p in patterns:
            # Do not keep a pattern if it has too low coverage
            if use_coverage_threshold and self.min_coverage_threshold is not None and p.coverage < self.min_coverage_threshold:
                continue
                
            # Do not select if no gain at all
            if p.metric <= 0:
                continue
                
            is_correlated = False
            for a in ans:
                if p.global_normalized_mutual_information(a) > self.correlation_threshold:
                    is_correlated = True
                    break
            if not is_correlated:
                ans.append(p)
                if len(ans) % int(k/10) == 0:
                    print(f"Finding top k: {len(ans)} / {k}")
                if len(ans) == k:
                    break     
        return ans
        
    def _select_alphabet_remove_others(self) -> [List[str], List[Pattern]]:
        assert self.candidate_alphabet is not None
        
        w_size = self.gaps_allowed if self.gaps_allowed is not None else self.window_size
        root = Pattern([], w_size, None, self)
        
        # Find information gain of each candidate
        canndidate_alphabet_patterns = []
        for c in tqdm(self.candidate_alphabet):
            w_size = self.gaps_allowed + 1 if self.gaps_allowed is not None else self.window_size
            the_candidate = Pattern([set([c])], w_size, root, self)
            canndidate_alphabet_patterns.append(the_candidate)
        
        # Find top k1 attributes to be the alphabet while removing correlated attributes
        alphabet_patterns = self._find_top_k_patterns(canndidate_alphabet_patterns, k = self.alphabet_size)
        alphabet = [list(p.pattern[0])[0] for p in alphabet_patterns]
        
        # Remove non-alphabet attributes from all_features of augmented texts
        for augtext in self.pos_augmented:
            augtext.keep_only_features(alphabet)
        for augtext in self.neg_augmented:
            augtext.keep_only_features(alphabet)
            
        return alphabet, alphabet_patterns
        
    
    def fit_transform(self, positives: List[Union[str, AugmentedText]], negatives: List[Union[str, AugmentedText]]) -> List[Pattern]:
        # Fit = Find a set of alphabet
        # 1. Create augmented texts
        print("Step 1: Create augmented texts")
        self.positives, self.negatives, self.pos_augmented, self.neg_augmented = [], [], [], []
        for is_neg, input_list in enumerate([positives, negatives]):
            for t in tqdm(input_list):
                if isinstance(t, str):
                    if not is_neg:
                        self.pos_augmented.append(AugmentedText(t, True, self.include_standard, self.include_custom))
                        self.positives.append(t)
                    else:
                        self.neg_augmented.append(AugmentedText(t, False, self.include_standard, self.include_custom))
                        self.negatives.append(t)
                else: # t is an AugmentedText
                    assert self.include_standard == t.include_standard
                    assert self.include_custom == t.include_custom
                    if not is_neg:
                        self.pos_augmented.append(t)
                        self.positives.append(t.text)
                    else:
                        self.neg_augmented.append(t)
                        self.negatives.append(t.text)
        
        # 2. Find frequent attributes (according to min_freq_threshold)
        print("Step 2: Find frequent attributes")
        self.candidate_alphabet = self._remove_nonfrequent_attributes()
        print(f"Total number of candidate alphabet = {len(self.candidate_alphabet)}, such as {self.candidate_alphabet[:5]}")
        
        # 3. Find alphabet set (according to alphabet_size and correlation_threshold)
        print("Step 3: Find alphabet set")
        self.alphabet, self.seed_patterns = self._select_alphabet_remove_others()
        print(f"Total number of alphabet = {len(self.alphabet)}")
        print(self.alphabet)
        print("Example of patterns")
        for p in self.seed_patterns[:5]:
            print(p)
        
        # 4. Grow and selecct patterns
        print("Step 4: Grow patterns")
        current_patterns = list(self.seed_patterns)
        last = list(current_patterns)
        visited = set([p.get_pattern_id() for p in current_patterns])
        for length in range(2, self.max_len+1):
            new_candidates = []
            for p in tqdm(last):
                for a in self.alphabet:
                    # Grow right
                    grow_right_candidate = p.pattern + [set([a])]
                    if Pattern.pattern_list2str(grow_right_candidate) not in visited:
                        w_size = self.gaps_allowed + len(grow_right_candidate) if self.gaps_allowed is not None else self.window_size
                        new_candidates.append(Pattern(grow_right_candidate, w_size, p, self))
                        visited.add(Pattern.pattern_list2str(grow_right_candidate))
                    # Grow inside
                    grow_inside_candidate = p.pattern[:-1] + [set([a]).union(p.pattern[-1])]
                    if Pattern.pattern_list2str(grow_inside_candidate) not in visited:
                        w_size = self.gaps_allowed + len(grow_inside_candidate) if self.gaps_allowed is not None else self.window_size
                        new_candidates.append(Pattern(grow_inside_candidate, w_size, p, self))
                        visited.add(Pattern.pattern_list2str(grow_inside_candidate))
            print(f'Length {length} / {self.max_len}; New candidates = {len(new_candidates)}')
            if len(new_candidates) == 0:
                break
            current_patterns = self._find_top_k_patterns(current_patterns + new_candidates, k = self.num_patterns)
            last = [p for p in current_patterns if p in new_candidates] # last is recently added patterns
            print("Example of current patterns")
            for p in current_patterns[:5]:
                print(p)
        self.extracted_patterns = current_patterns
        return current_patterns

    def to_json(self, filepath: str):
        # Rules
        rules = []
        for idx, p in enumerate(self.extracted_patterns):
            rules.append({
                'index': idx,
                'pattern': p.get_pattern_id(),
                'meaning': pattern2text(p),
                '#pos': p.pos_example_labels.count(True), 
                '#neg': p.neg_example_labels.count(True),
                'score': p.metric,
                'coverage': p.coverage, 
                'precision': p.precision, 
                'recall': p.recall, 
                'F1': (2 * p.precision * p.recall) / (p.precision + p.recall) if (p.precision is not None) else None
                })

        # Configuration
        configuration = {
            'min_freq_threshold': self.min_freq_threshold,
            'correlation_threshold': self.correlation_threshold,
            'alphabet_size': self.alphabet_size,
            'num_patterns': self.num_patterns,
            'max_len': self.max_len,
            'window_size': self.window_size,
            'gaps_allowed': self.gaps_allowed,
            'gain_criteria': str(self.gain_criteria),
            'min_coverage_threshold': self.min_coverage_threshold,
            'include_standard': self.include_standard, 
            'include_custom': [attr.name for attr in self.include_custom]
        } 

        main_obj = {
            'configuration': configuration,
            'alphabet': self.alphabet,
            'rules': rules
        }
        
        with open(filepath, "w") as f: # Write a JSON file
            json.dump(main_obj, f)

        print(f'Successfully dump the results to {filepath}')

    def to_csv(self, filepath: str):
        patterns2csv(self.extracted_patterns, filepath)


# ========== Simplify pattern set ==========
def is_specialized(p1:Pattern, p2:Pattern) -> bool: # Return True if p1 is a specialization of p2 including the case where p1_labels == p2_labels
    p1_labels = [True if v else False for v in p1.pos_example_labels + p1.neg_example_labels]
    p2_labels = [True if v else False for v in p2.pos_example_labels + p2.neg_example_labels]
    for idx, v in enumerate(p1_labels):
        if v:
            if not p2_labels[idx]:
                return False
    return True

def remove_specialized_patterns(patterns: List[Pattern],
                                mode: int = 1,
                                metric: Optional[Union[Callable, str]] = None
    ) -> List[Pattern]:
    # Mode = 1: Remove pattern p2 if there exists p1 in the patterns set such that p2 is a specialization of p1 and metric of p2 is lower than p1
    # Mode = 2: Remove pattern p2 if there exists p1 in the patterns set such that p2 is a specialization of p1 regardless of the metric value of p1 and p2
    
    # Find the metric function
    if callable(metric):
        metric_func = metric
    elif metric is None:
        metric_func = lambda x: x.metric
    elif metric == 'global':
        metric_func = lambda x: x.global_information_gain 
    elif metric == 'local':
        metric_func = lambda x: x.information_gain 
    elif metric == 'relative':
        metric_func = lambda x: x.relative_information_gain
    elif metric.startswith('F_'):
        try:
            beta = float(metric[2:])
        except:
            assert False, f"Invalid metric: {metric}"
        metric_func = lambda x: (1+beta**2) * (x.precision * x.recall) / ((x.precision*beta**2) + x.recall) if (x.precision is not None) else 0.0
    else:
        assert False, f"Invalid metric {metric}"

    # Remove patterns
    the_list = patterns
    to_remove = set() 
    for idx1, p in enumerate(the_list):
        for idx2, q in enumerate(the_list):
            if idx1 == idx2:
                continue
            if mode == 1 and metric_func(p) >= metric_func(q) and is_specialized(q, p):
                to_remove.add(idx2)
            elif mode == 2 and is_specialized(q, p):
                to_remove.add(idx2)
    return [p for idx, p in enumerate(the_list) if idx not in to_remove]

# ========== Translation ==========
def attr2score(attr: str) -> float:
    # Return a score used to rank the attribute during translation
    attr_type = attr.split(':')[0]
    if attr_type == 'SPACY':
        attr_type = attr.split('-')[0]

    SCORES = {
        'TEXT': 0,
        'SPACY:NER': 10,
        'SPACY:POS': 20,
        'SENTIMENT': 30,
        'HYPERNYM': 40,
        'SPACY:DEP': 50,
    }

    return SCORES.get(attr_type, 100)

def attr2text(attr: str, p: Pattern, is_complement: bool = False) -> str:
    attr_type = attr.split(':')[0]
    if attr_type in p.all_attributes_dict:
        return p.all_attributes_dict[attr_type].translation_function(attr, is_complement)
    else:
        return default_attr2text(attr, is_complement)

def slot2text(s: Set[str], p: Pattern) -> str:
    s = list(s)
    s.sort(key = lambda x: attr2score(x))
    ans = ''
    for aidx, a in enumerate(s):
        if aidx == 0:
            ans += attr2text(a, p, is_complement = False)
        elif aidx == 1:
            ans += f' which is also {attr2text(a, p, is_complement = True)}'
        elif aidx == len(s)-1:
            if aidx == 2:
                ans += f' and {attr2text(a, p, is_complement = True)}'
            else:
                ans += f', and {attr2text(a, p, is_complement = True)}'
        else:
            ans += f', {attr2text(a, p, is_complement = True)}'
    return ans

def pattern2text(p: Pattern) -> str:
    has_gap = p.window_size > len(p.pattern)
    connector = 'immediately' if not has_gap else 'closely'
    ans = ''
    for sidx, slot in enumerate(p.pattern):
        if sidx == 0:
            ans += slot2text(slot, p)
        elif sidx == 1:
            ans += f', {connector} followed by {slot2text(slot, p)}'
        else:
            # ans += f', and then {connector} followed by {slot2text(slot, p)}'
            ans += f', and then by {slot2text(slot, p)}'
    return ans[0].upper() + ans[1:]

# ========== Feature extraction ==========

def extract_features(texts: List[str], 
                     patterns: List[Pattern],
                     polarity: bool = False, # If True, features will be [1, 0, -1] where 1 means positive rule match and -1 means negative rule match
                     include_standard: List[str] = ['TEXT', 'POS', 'DEP', 'NER', 'HYPERNYM', 'SENTIMENT'], 
                     include_custom: List[CustomAttribute] = []): # Return numpy array [len(texts), len(patterns)]
    if polarity:
        for p in patterns:
            assert p.support_class in ["Positive", "Negative"], "Cannot set Polarity = True as some of the patterns do not have a valid support_class"

    ans = []
    for t in tqdm(texts):
        vector = []
        augtext = AugmentedText(t, None, include_standard, include_custom)
        for p in patterns:
            if p.is_match(augtext)[0]:
                if polarity:
                    value = 1 if p.support_class == 'Positive' else -1
                    vector.append(value)
                else:
                    vector.append(1)
            else:
                vector.append(0)
        ans.append(vector)
    return np.array(ans)            

# ========== Export ==========
def patterns2csv(patterns: List[Pattern],
                 filepath: str):

    data = []
    for idx, p in enumerate(patterns):
        data.append({
            'index': idx,
            'pattern': p.get_pattern_id(),
            'meaning': pattern2text(p),
            '#pos': p.pos_example_labels.count(True), 
            '#neg': p.neg_example_labels.count(True),
            'score': p.metric,
            'coverage': p.coverage, 
            'precision': p.precision, 
            'recall': p.recall, 
            'F1': (2 * p.precision * p.recall) / (p.precision + p.recall) if (p.precision is not None) else None
            })
    with open(filepath, 'w', newline='\n', encoding='utf-8') as csv_file:
        fieldnames = ['index', 'pattern', 'meaning', '#pos', '#neg', 'score', 'coverage', 'precision', 'recall', 'F1']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f'Successful: {len(data)} rules have been written to {filepath}')
       
# ========== Main ==========

if __name__ == "__main__":
    print("Running GrASP ...")
    print(AugmentedText("London is the capital and largest city of England and the United Kingdom."))
    print(AugmentedText("This was the worst restaurant I have ever had the misfortune of eating at."))