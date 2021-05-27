# GrASP
**GrASP** (GReedy Augmented Sequential Patterns) is an algorithm for extracting patterns from text data ([Shnarch et. al., 2017](https://www.aclweb.org/anthology/D17-1140.pdf)). Basically, it takes as input a list of positive and negative examples of a target phenomenon and outputs a ranked list of patterns that distinguish between the positive and the negative examples. For instance, two GrASP patterns from two use cases are shown in the Table below along with the sentences they match.

![Examples of GrASP patterns and the examples they match](figs/patterns.PNG)

This repository provides the implementation of GrASP, a web-based tool for exploring the results from GrASP, and two example notebooks for use cases of GrASP. This project is a joint collaboration between Imperial College London and IBM Research.

**Paper**: [GrASP: A Library for Extracting and Exploring Human-Interpretable Textual Patterns](https://arxiv.org/abs/2104.03958)

**Authors**: [Piyawat Lertvittayakumjorn](https://www.doc.ic.ac.uk/~pl1515/), [Leshem Choshen](https://ktilana.wixsite.com/leshem-choshen), [Eyal Shnarch](https://researcher.watson.ibm.com/researcher/view.php?person=il-EYALS), and [Francesca Toni](https://www.doc.ic.ac.uk/~ft/). 

**Contact**: Piyawat Lertvittayakumjorn (pl1515 [at] imperial [dot] ac [dot] uk)

## Requirements

### For the GrASP library
- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- Required packages
    - [numpy](https://numpy.org/)==1.16.3
    - [scikit-learn](https://scikit-learn.org/stable/)==0.23.2
    - [nltk](https://www.nltk.org/)==3.2.4
    - [spacy](https://spacy.io/)==2.0.12 (en_core_web_sm)
    - [termcolor](https://pypi.org/project/termcolor/)==1.1.0
    - [tqdm](https://pypi.org/project/tqdm/)==4.46.0

### For the web-based exploration tool
- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- Required packages
    - [Flask](https://flask.palletsprojects.com/)==0.12.2

Note that the packages with slightly different versions might work as well.

## Installation

1. Clone this repository 
2. Download the required packages listed above. Or you may use the `requirements.txt` file to download all of them by running the following command inside the cloned repository.

        pip install -r requirements.txt
3. Run the following commands to download required resources.

        python -m spacy download en_core_web_sm
        python -c "import nltk; nltk.download('wordnet');"



## Usage

```python    
import grasp
# Step 1: Create the GrASP model
grasp_model = grasp.GrASP(num_patterns = 200, 
                    gaps_allowed = 2, 
                    alphabet_size = 200, 
                    include_standard = ['TEXT', 'POS', 'NER', 'SENTIMENT'])
# Step 2: Fit it to the training data
the_patterns = grasp_model.fit_transform(pos_exs, neg_exs)
# Step 3: Export the results 
grasp_model.to_csv('results.csv')
grasp_model.to_json('results.json')
```

As shown above, GrASP can be used in three steps:
1. Creating a GrASP model (with hyperparameters specified)
2. Fit the GrASP model to the lists of positive and negative examples
3. Export the results to a csv or a json file

### Hyperparameters for GrASP (Step 1)
- `min_freq_threshold` (float, default = 0.005) -- Attributes which appear less often than this proportion of the number of training examples will be discarded as they are non-frequent.
- `correlation_threshold` (float, default = 0.5) -- Attributes/patterns whose correlation to some previously selected attribute/pattern is above this threshold, measured by the normalized mutual information, will be discarded.
- `alphabet_size` (int, default = 100) -- The alphabet size.
- `num_patterns` (int, default = 100) -- The number of output patterns.
- `max_len` (int, default = 5) -- The maximum number of attributes per pattern.
- `window_size` (Optional[int], default = 10) -- The window size for the output patterns.
- `gaps_allowed` (Optional[int], default = None) -- If gaps allowed is not None, it overrules the window size and specifies the number of gaps allowed in each output pattern.
- `gain_criteria` (str or Callable[[Pattern], float]], default = 'global') -- The criterion for selecting alphabet and patterns. 'global' refers to the information gain criterion. The current version also supports a criterion of `F_x` (such as `F_0.01`). 
- `min_coverage_threshold` (Optional[float], default = None) -- The minimum proportion of examples matched for output patterns (so GrASP does not generate too specific patterns).
- `print_examples` (Union[int, Sequence[int]], default = 2) -- The number of examples and counter-examples to print when printing a pattern. If `print_examples` equals `(x, y)`, it prints `x` examples and `y` counter-examples for each pattern. If `print_examples` equals `x`, it is equivalent to `(x, x)`.
- `include_standard` (List[str], default = ['TEXT', 'POS', 'NER', 'HYPERNYM', 'SENTIMENT']) -- The built-in attributes to use. Available options are ['TEXT', 'LEMMA', 'POS', 'DEP', 'NER', 'HYPERNYM', 'SENTIMENT'].
- `include_custom` (List[CustomAttribute], default = []) -- The list of custom attributes to use.

### Built-in attributes
The current implementation of GrASP consists of seven standard attributes. The full lists of tags for POS, DEP, and NER can be found from [SPACY](https://github.com/explosion/spaCy/blob/master/spacy/glossary.py).

1. **TEXT attribute** of a token is the token in lower case.
2. **LEMMA attribute** of a token is its lemma obtained from SPACY.
3. **POS attribute** of a token is the part-of-speech tag of the token according to [the universal POS tags](https://universaldependencies.org/u/pos/)
4. **DEP attribute** of a token is the dependency parsing tag of the token (the type of syntactic relation that connects the child to the head)
5. **NER attribute** is a token (if any) is the named entity type of the token.
6. **HYPERNYM attribute** of a/an (noun, verb, adjective, adverb) token is the synsets of the hypernyms of the token (including the synset of the token itself). The hypernym hierarchy is based on [WordNet (nltk)](https://www.nltk.org/howto/wordnet.html). Note that we consider only **three levels of synsets** above the token of interest in order to exclude synsets that are too abstract to comprehend (e.g., psychological feature, group action, and entity).
7. **SENTIMENT attribute** of a token (if any) indicates the sentiment (pos or neg) of the token based on the lexicon in [Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In International Conference on Knowledge Discovery and Data Mining, KDD’04, pages 168–177.](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)

### Examples of augmented texts

Input sentence: London is the capital and largest city of England and the United Kingdom.

```
London: {'SPACY:NER-GPE', 'SPACY:POS-PROPN', 'TEXT:london', 'HYPERNYM:london.n.01'}
is: {'SPACY:POS-VERB', 'TEXT:is', 'HYPERNYM:be.v.01'}
the: {'TEXT:the', 'SPACY:POS-DET'}
capital: {'SPACY:POS-NOUN', 'HYPERNYM:capital.n.06', 'TEXT:capital'}
and: {'TEXT:and', 'SPACY:POS-CCONJ'}
largest: {'HYPERNYM:large.a.01', 'TEXT:largest', 'SPACY:POS-ADJ'}
city: {'HYPERNYM:urban_area.n.01', 'HYPERNYM:municipality.n.01', 'TEXT:city', 'SPACY:POS-NOUN', 'HYPERNYM:geographical_area.n.01', 'HYPERNYM:administrative_district.n.01', 'HYPERNYM:district.n.01', 'HYPERNYM:city.n.01'}
of: {'SPACY:POS-ADP', 'TEXT:of'}
England: {'SPACY:NER-GPE', 'HYPERNYM:england.n.01', 'SPACY:POS-PROPN', 'TEXT:england'}
and: {'TEXT:and', 'SPACY:POS-CCONJ'}
the: {'TEXT:the', 'SPACY:POS-DET', 'SPACY:NER-GPE'}
United: {'TEXT:united', 'SPACY:NER-GPE', 'SPACY:POS-PROPN'}
Kingdom: {'HYPERNYM:kingdom.n.05', 'SPACY:POS-PROPN', 'HYPERNYM:taxonomic_group.n.01', 'TEXT:kingdom', 'SPACY:NER-GPE', 'HYPERNYM:biological_group.n.01', 'HYPERNYM:group.n.01'}
.: {'TEXT:.', 'SPACY:POS-PUNCT'}
```

Input sentence: This was the worst restaurant I have ever had the misfortune of eating at.

```
This: {'SPACY:POS-DET', 'TEXT:this'}
was: {'TEXT:was', 'SPACY:POS-VERB', 'HYPERNYM:be.v.01'}
the: {'TEXT:the', 'SPACY:POS-DET'}
worst: {'SENTIMENT:neg', 'HYPERNYM:worst.a.01', 'SPACY:POS-ADJ', 'TEXT:worst'}
restaurant: {'SPACY:POS-NOUN', 'HYPERNYM:artifact.n.01', 'HYPERNYM:restaurant.n.01', 'HYPERNYM:building.n.01', 'TEXT:restaurant', 'HYPERNYM:structure.n.01'}
I: {'TEXT:i', 'SPACY:POS-PRON'}
have: {'SPACY:POS-VERB', 'TEXT:have', 'HYPERNYM:own.v.01'}
ever: {'SPACY:POS-ADV', 'TEXT:ever', 'HYPERNYM:always.r.01'}
had: {'SPACY:POS-VERB', 'TEXT:had', 'HYPERNYM:own.v.01'}
the: {'TEXT:the', 'SPACY:POS-DET'}
misfortune: {'TEXT:misfortune', 'HYPERNYM:fortune.n.04', 'HYPERNYM:state.n.02', 'SPACY:POS-NOUN', 'SENTIMENT:neg', 'HYPERNYM:misfortune.n.02', 'HYPERNYM:condition.n.03'}
of: {'SPACY:POS-ADP', 'TEXT:of'}
eating: {'SPACY:POS-VERB', 'HYPERNYM:change.v.01', 'HYPERNYM:damage.v.01', 'TEXT:eating', 'HYPERNYM:corrode.v.01'}
at: {'SPACY:POS-ADP', 'TEXT:at'}
.: {'TEXT:.', 'SPACY:POS-PUNCT'}
```
### Supporting features

- **Translating from a pattern to its English explanation**

```python
# Continue from the code snippet above
print(grasp.pattern2text(the_patterns[0]))
```
- **Removing redundant patterns**
    - Mode = 1: Remove pattern p2 if there exists p1 in the patterns set such that p2 is a specialization of p1 and metric of p2 is lower than p1
    - Mode = 2: Remove pattern p2 if there exists p1 in the patterns set such that p2 is a specialization of p1 regardless of the metric value of p1 and p2

```python
selected_patterns = grasp.remove_specialized_patterns(the_patterns, metric = lambda x: x.precision, mode = 1)
```
- **Vectorizing texts using patterns**

```python
X_array = grasp.extract_features(texts = pos_exs + neg_exs,
                 patterns = selected_patterns,
                 include_standard = ['TEXT', 'POS', 'NER', 'SENTIMENT'])
```

- **Creating a custom attribute**

```python
ARGUMENTATIVE_LEXICON = [line.strip().lower() for line in open('data/argumentative_unigrams_lexicon_shortlist.txt', 'r') if line.strip() != '']
def _argumentative_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
    tokens = map(str.lower, tokens)
    ans = []
    for t in tokens:
        t_ans = []
        if t.lower() in ARGUMENTATIVE_LEXICON:
            t_ans.append('Yes')
        ans.append(set(t_ans))
    return ans

def _argumentative_translation(attr:str, 
                      is_complement:bool = False) -> str:
    word = attr.split(':')[1]
    assert word == 'Yes'
    return 'an argumentative word'

ArgumentativeAttribute = grasp.CustomAttribute(name = 'ARGUMENTATIVE', 
    extraction_function = _argumentative_extraction, 
    translation_function = _argumentative_translation)

grasp_model = grasp.GrASP(include_standard = ['TEXT', 'POS', 'NER', 'SENTIMENT'],
                          include_custom = [ArgumentativeAttribute]
                         )
```

## The Web Exploration Tool

**Requirements**: Python 3.6 and Flask

**Steps**
1. To import json result files to the web system, please edit `web_demo/settings.py`

```python
CASES = {
    1: {'name': 'SMS Spam Classification', 'result_path': '../results/case_study_1.json'},
    2: {'name': 'Topic-dependent Argument Mining', 'result_path': '../results/case_study_2.json'},
}
```

2. To run the web system, go inside the web_demo folder and run `python -u app.py`. You will see the following messages.

```
$ python -u app.py
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 553-838-653
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

```

So, using your web browser, you can access all the reports at http://127.0.0.1:5000/.

Note that we have the live demo of our two case studies (spam detection and argument mining) now running [here](https://plkumjorn.pythonanywhere.com/). 

## Repository Structure

    .
    ├── data/               # For downloaded data
    ├── figs/               # For figures used in this README file
    ├── resources/          # For resources for built-in attributes
    │   └── opinion-lexicon-English/    # Lexicon for the sentiment attributes
    ├── results/            # For exported results (.json, .csv)
    ├── web_demo/           # The web-based exploration tool
    │    ├── static/        # For CSS and JS files
    │    ├── templates/     # For Jinja2 templates for rendering the html output 
    │    ├── app.py         # The main Flask application
    │    └── settings.py    # For specifying locations of JSON result files to explore   
    ├── .gitignore
    ├── CaseStudy1_SMSSpamCollection.ipynb
    ├── CaseStudy2_ArgumentMining.ipynb
    ├── LICENSE
    ├── README.md
    └── grasp.py            # The main grasp code


## Citation

If you use or refer to the implementation in this repository, please cite the following paper.

    @misc{lertvittayakumjorn2021grasp,
        title={GrASP: A Library for Extracting and Exploring Human-Interpretable Textual Patterns}, 
        author={Piyawat Lertvittayakumjorn and Leshem Choshen and Eyal Shnarch and Francesca Toni},
        year={2021},
        eprint={2104.03958},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

If you refer to [the original GrASP algorithm](https://www.aclweb.org/anthology/D17-1140.pdf), please cite the following paper.

    @inproceedings{shnarch-etal-2017-grasp,
        title = "{GRASP}: Rich Patterns for Argumentation Mining",
        author = "Shnarch, Eyal  and
          Levy, Ran  and
          Raykar, Vikas  and
          Slonim, Noam",
        booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
        month = sep,
        year = "2017",
        address = "Copenhagen, Denmark",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/D17-1140",
        doi = "10.18653/v1/D17-1140",
        pages = "1345--1350",
    }


## Contact
Piyawat Lertvittayakumjorn (pl1515 [at] imperial [dot] ac [dot] uk)