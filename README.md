# GrASP
An implementation of GrASP (Shnarch et. al., 2017). Note that this repository is **under construction**.

## Attributes

The current implementation of GrASP consists of five standard attributes (See line 61-144 in grasp.py). The full lists of tags for POS, DEP, and NER are from [SPACY](https://github.com/explosion/spaCy/blob/master/spacy/glossary.py).

1. **TEXT attribute** of a token is the token in lower case.
2. **POS attribute** of a token is the part-of-speech tag of the token according to [the universal POS tags](https://universaldependencies.org/u/pos/)

```
GLOSSARY = {
    # POS tags
    # Universal POS Tags
    # http://universaldependencies.org/u/pos/
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CONJ": "conjunction",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
    "EOL": "end of line",
    "SPACE": "space",
}
```

3. **DEP attribute** of a token is the dependency parsing tag of the token (the type of syntactic relation that connects the child to the head)

```
GLOSSARY = {
    # Dependency Labels (English)
    # ClearNLP / Universal Dependencies
    # https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
    "acl": "clausal modifier of noun (adjectival clause)",
    "acomp": "adjectival complement",
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "agent": "agent",
    "amod": "adjectival modifier",
    "appos": "appositional modifier",
    "attr": "attribute",
    "aux": "auxiliary",
    "auxpass": "auxiliary (passive)",
    "case": "case marking",
    "cc": "coordinating conjunction",
    "ccomp": "clausal complement",
    "clf": "classifier",
    "complm": "complementizer",
    "compound": "compound",
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "csubjpass": "clausal subject (passive)",
    "dative": "dative",
    "dep": "unclassified dependent",
    "det": "determiner",
    "discourse": "discourse element",
    "dislocated": "dislocated elements",
    "dobj": "direct object",
    "expl": "expletive",
    "fixed": "fixed multiword expression",
    "flat": "flat multiword expression",
    "goeswith": "goes with",
    "hmod": "modifier in hyphenation",
    "hyph": "hyphen",
    "infmod": "infinitival modifier",
    "intj": "interjection",
    "iobj": "indirect object",
    "list": "list",
    "mark": "marker",
    "meta": "meta modifier",
    "neg": "negation modifier",
    "nmod": "modifier of nominal",
    "nn": "noun compound modifier",
    "npadvmod": "noun phrase as adverbial modifier",
    "nsubj": "nominal subject",
    "nsubjpass": "nominal subject (passive)",
    "nounmod": "modifier of nominal",
    "npmod": "noun phrase as adverbial modifier",
    "num": "number modifier",
    "number": "number compound modifier",
    "nummod": "numeric modifier",
    "oprd": "object predicate",
    "obj": "object",
    "obl": "oblique nominal",
    "orphan": "orphan",
    "parataxis": "parataxis",
    "partmod": "participal modifier",
    "pcomp": "complement of preposition",
    "pobj": "object of preposition",
    "poss": "possession modifier",
    "possessive": "possessive modifier",
    "preconj": "pre-correlative conjunction",
    "prep": "prepositional modifier",
    "prt": "particle",
    "punct": "punctuation",
    "quantmod": "modifier of quantifier",
    "rcmod": "relative clause modifier",
    "relcl": "relative clause modifier",
    "reparandum": "overridden disfluency",
    "root": "root",
    "vocative": "vocative",
    "xcomp": "open clausal complement",
}
```

4. **NER attribute** is a token (if any) is the named entity type of the token.

```
GLOSSARY = {
    # Named Entity Recognition
    # OntoNotes 5
    # https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf
    "PERSON": "People, including fictional",
    "NORP": "Nationalities or religious or political groups",
    "FACILITY": "Buildings, airports, highways, bridges, etc.",
    "FAC": "Buildings, airports, highways, bridges, etc.",
    "ORG": "Companies, agencies, institutions, etc.",
    "GPE": "Countries, cities, states",
    "LOC": "Non-GPE locations, mountain ranges, bodies of water",
    "PRODUCT": "Objects, vehicles, foods, etc. (not services)",
    "EVENT": "Named hurricanes, battles, wars, sports events, etc.",
    "WORK_OF_ART": "Titles of books, songs, etc.",
    "LAW": "Named documents made into laws.",
    "LANGUAGE": "Any named language",
    "DATE": "Absolute or relative dates or periods",
    "TIME": "Times smaller than a day",
    "PERCENT": 'Percentage, including "%"',
    "MONEY": "Monetary values, including unit",
    "QUANTITY": "Measurements, as of weight or distance",
    "ORDINAL": '"first", "second", etc.',
    "CARDINAL": "Numerals that do not fall under another type",
}
``` 

5. **HYPERNYM attribute** of a/an (noun, verb, adjective, adverb) token is all the synsets of the hypernyms of the token (including the synset of the token itself). The hypernym hierarchy is based on [WordNet (nltk)](https://www.nltk.org/howto/wordnet.html).

6. **SENTIMENT attribute** of a token (if any) indicates the sentiment (pos or neg) of the token based on the lexicon in [Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In International Conference on Knowledge Discovery and Data Mining, KDD’04, pages 168–177.](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)

### An example of augmented texts

Input sentence: London is the capital and largest city of England and the United Kingdom.

```
London: {'TEXT:london', 'HYPERNYM:london.n.01', 'SPACY:DEP-nsubj', 'SPACY:NER-GPE', 'SPACY:POS-PROPN'}
is: {'TEXT:is', 'SPACY:POS-VERB', 'SPACY:DEP-ROOT', 'HYPERNYM:be.v.01'}
the: {'SPACY:DEP-det', 'SPACY:POS-DET', 'TEXT:the'}
capital: {'TEXT:capital', 'SPACY:DEP-attr', 'HYPERNYM:capital.n.06', 'SPACY:POS-NOUN'}
and: {'TEXT:and', 'SPACY:POS-CCONJ', 'SPACY:DEP-cc'}
largest: {'HYPERNYM:large.a.01', 'TEXT:largest', 'SPACY:DEP-amod', 'SPACY:POS-ADJ'}
city: {'HYPERNYM:region.n.03', 'HYPERNYM:district.n.01', 'HYPERNYM:location.n.01', 'SPACY:POS-NOUN', 'HYPERNYM:entity.n.01', 'HYPERNYM:municipality.n.01', 'SPACY:DEP-conj', 'HYPERNYM:urban_area.n.01', 'HYPERNYM:physical_entity.n.01', 'HYPERNYM:administrative_district.n.01', 'TEXT:city', 'HYPERNYM:city.n.01', 'HYPERNYM:geographical_area.n.01', 'HYPERNYM:object.n.01'}
of: {'SPACY:POS-ADP', 'TEXT:of', 'SPACY:DEP-prep'}
England: {'TEXT:england', 'HYPERNYM:england.n.01', 'SPACY:DEP-pobj', 'SPACY:NER-GPE', 'SPACY:POS-PROPN'}
and: {'TEXT:and', 'SPACY:POS-CCONJ', 'SPACY:DEP-cc'}
the: {'SPACY:DEP-det', 'SPACY:POS-DET', 'TEXT:the', 'SPACY:NER-GPE'}
United: {'SPACY:DEP-compound', 'SPACY:POS-PROPN', 'SPACY:NER-GPE', 'TEXT:united'}
Kingdom: {'HYPERNYM:biological_group.n.01', 'HYPERNYM:kingdom.n.05', 'SPACY:NER-GPE', 'SPACY:POS-PROPN', 'HYPERNYM:entity.n.01', 'SPACY:DEP-conj', 'TEXT:kingdom', 'HYPERNYM:abstraction.n.06', 'HYPERNYM:group.n.01', 'HYPERNYM:taxonomic_group.n.01'}
.: {'SPACY:POS-PUNCT', 'SPACY:DEP-punct', 'TEXT:.'}
```