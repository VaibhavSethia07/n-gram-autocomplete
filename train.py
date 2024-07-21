import math
import random
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer

nltk.download("punkt")

dataset = load_dataset("vicgalle/alpaca-gpt4")
df = pd.DataFrame(dataset["train"])
df["instruction_output"] = df[["instruction", "output"]].apply(lambda x: ' '.join(x), axis=1)


@dataclass
class Tokenizer:
    """Tokenize the text into word-tokenized sentences
        Args:
            source pandas.Series: Raw text series
    """
    source: pd.Series
    _sentences: List[str] = field(default_factory=list)

    @property
    def sentences(self):
        """The data split into sentences"""
        if not self._sentences:
            self._sentences = self.get_tokenized_data(data=self.source)
        return self._sentences

    def tokenize_text_to_sentences(self, data: pd.Series) -> List[str]:
        """
        Split data by line break "\n", sentence completion tokens into sentences
        Args:
            data pandas.Series: Raw text series
        Returns:
            sentences List[List[str]]: List of sentences
        """
        pst = PunktSentenceTokenizer()
        sentences = list()
        for text in data.values:

            # Tokenize text into sentences
            text_sentences = pst.tokenize(text=text)
            sentences.extend(text_sentences)

        return sentences

    def clean_words(self, words: List[str]) -> List[str]:
        """
        Clean the words by removing punctuations, and numeric tokens
            Args:
                words List[str]: Raw words
            Returns:
                cleaned_words List[str] Words without punctuations, and numeric tokens
        """
        cleaned_words = list()
        for word in words:
            # Skip punctuations
            if word in string.punctuation:
                continue

            # Skip numbers
            if word.isnumeric():
                continue

            cleaned_words.append(word.lower())
        return cleaned_words

    def tokenize_sentences(self, sentences: List[str]) -> List[List[str]]:
        """
        Tokenize sentences into tokens (words)
        Args:
            sentences List[str]: List of sentences
        Returns:
            tokenized_sentences List[List[str]]: List of tokenized sentences
        """
        tokenized_sentences = list()

        for sentence in sentences:
            # Tokenize sentence into words
            words = word_tokenize(text=sentence)
            cleaned_words = self.clean_words(words=words)
            if len(cleaned_words) == 0:
                continue
            tokenized_sentences.append(cleaned_words)

        return tokenized_sentences

    def get_tokenized_data(self, data: pd.Series) -> List[List[str]]:
        """
        Make a list of tokenized sentences
        Args:
            data pandas.Series: Raw text
        Returns:
            tokens List[List[str]]: List of tokenized sentences
        """
        sentences = self.tokenize_text_to_sentences(data=data)
        tokens = self.tokenize_sentences(sentences=sentences)

        return tokens


@dataclass
class TrainTestSplit:
    """Splits the data into train and test sets
        Args:
            data List[List[str]]: List of tokenized sentences
            seed int: Random seed value
            training_fraction float: Fraction of data to be used for training
    """
    data: List[List[str]]
    seed: int = 7
    training_fraction: float = 1
    _shuffled: List[List[str]] = field(default_factory=list)
    _training: List[List[str]] = field(default_factory=list)
    _testing: List[List[str]] = field(default_factory=list)
    _split: int = field(default_factory=int)

    @property
    def shuffled(self):
        """Shuffled data"""
        if not self._shuffled:
            random.seed(self.seed)
            # Don't use random.shuffle because it is in-place shuffle
            self._shuffled = random.sample(population=self.data, k=len(self.data))
        return self._shuffled

    @property
    def split(self):
        """Slice value for training and testing"""
        if not self._split:
            self._split = int(len(self.data)*self.training_fraction)
        return self._split

    @property
    def training(self):
        """Training set"""
        if not self._training:
            self._training = self.shuffled[0:self.split]
        return self._training

    @property
    def testing(self):
        """Testing set"""
        if not self._testing:
            self._testing = self.shuffled[self.split:]
        return self._testing


@dataclass
class CountProcessor:
    """Processes the data to have unknowns
        Args:
            training List[List[str]]: List of tokenized training sentences
            testing List[List[str]]: List of tokenized testing sentences
            threshold int: Minimum number of occurences for a word to be in closed vocabulary
            unknown_token str: Unknown token
    """
    training: List[List[str]]
    testing: List[List[str]]
    threshold: int = 2
    unknown_token: str = "<unk>"
    _vocabulary: Dict[str, int] = field(default_factory=dict)
    _closed_vocabulary_set: Set[str] = field(default_factory=set)
    _training_data_unknowns: List[List[str]] = field(default_factory=list)
    _testing_data_unknowns: List[List[str]] = field(default_factory=list)

    @property
    def vocabulary(self):
        """The tokens in the training set and their frequency"""
        if not self._vocabulary:
            self._vocabulary = self.create_vocabulary()
        return self._vocabulary

    @property
    def closed_vocabulary(self):
        """The tokens in the training set that appear more than the `threshold` times"""
        if not self._closed_vocabulary_set:
            self._closed_vocabulary_set = self.create_closed_vocabulary_set()
        return self._closed_vocabulary_set

    @property
    def training_data_unknowns(self):
        """Training data with words below threshold replaced with `unknown_token`"""
        if not self._training_data_unknowns:
            self._training_data_unknowns = self.replace_oov_words_by_unknown_token(self.training)
        return self._training_data_unknowns

    @property
    def testing_data_unknowns(self):
        """Testing data with words below threshold replaced with `unknown_token`"""
        if not self._testing_data_unknowns:
            self._testing_data_unknowns = self.replace_oov_words_by_unknown_token(self.testing)
        return self._testing_data_unknowns

    def create_vocabulary(self) -> Dict[str, int]:
        """
        Count the number of word appearances in tokenized sentences
            Returns:
                vocabulary Dict[str, int]: Dictionary that maps word(str) to its frequency(int)
        """
        vocabulary = defaultdict(int)
        for sentence in self.training:
            for word in sentence:
                vocabulary[word] += 1

        return dict(vocabulary)

    def create_closed_vocabulary_set(self) -> Dict[str, int]:
        """
        Find the words that appear more than the threshold frequency
            Returns:
                closed_vocabulary Set[str]: Set of words that appear `threshold` or more times
        """
        closed_vocabulary = set()
        for word, freq in self.vocabulary.items():
            if freq >= self.threshold:
                closed_vocabulary.add(word)
        return closed_vocabulary

    def replace_oov_words_by_unknown_token(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Replace words not in the given vocabulary with '<unk>' token.
            Args:
                sentences List[List[str]]: List of word tokenized sentences
            Returns:
                replaced_tokenized_sentences List[List[str]]:  List of word tokenized sentences with out-of-vocabulary words converted to `unknown_token`
        """
        replaced_tokenized_sentences = list()

        for sentence in sentences:
            replaced_sentence = list()
            for word in sentence:
                if word not in self.closed_vocabulary:
                    replaced_sentence.append(self.unknown_token)
                else:
                    replaced_sentence.append(word)

            replaced_tokenized_sentences.append(replaced_sentence)

        return replaced_tokenized_sentences


@dataclass
class NGrams:
    """
    The N-Gram Language Model
        Args:
            data List[List[str]]: List of tokenized sentences
            n int: Size of the N-Gram
            start_token str: Start of sentence token
            end_token str: End of sentence token
    """
    data: List[List[str]]
    n: int
    start_token: str = "<s>"
    end_token: str = "<e>"
    _start_tokens: List[str] = field(default_factory=list)
    _end_tokens: List[str] = field(default_factory=list)
    _sentences: List[List[str]] = field(default_factory=list)
    _n_grams: List[List[str]] = field(default_factory=list)
    _counts: Dict[Tuple[str], int] = field(default_factory=Counter)

    @property
    def start_tokens(self):
        """List of `n` start tokens"""
        if not self._start_tokens:
            self._start_tokens = [self.start_token]*self.n
        return self._start_tokens

    @property
    def end_tokens(self):
        """List of 1 end tokens"""
        if not self._end_tokens:
            self._end_tokens = [self.end_token]
        return self._end_tokens

    @property
    def sentences(self):
        """The data augmented with start and end tokens and converted to tuples"""
        if not self._sentences:
            self._sentences = [tuple(self.start_tokens + sentence + self.end_tokens) for sentence in self.data]
        return self._sentences

    @property
    def n_grams(self) -> List[str]:
        """The n-grams from the data
        Warning:This method flattens the n-grams so there isn't any sentence structure
        """
        if not self._n_grams:
            self._n_grams = chain.from_iterable([
                [sentence[cut:cut+self.n] for cut in range(0, len(sentence) - self.n + 1)]
                for sentence in self.sentences])
        return self._n_grams

    @property
    def counts(self) -> Counter:
        """Count of all n-grams in the data
        Returns: A dictionary that maps a tuple of n-words to its frequency
        """
        if not self._counts:
            self._counts = Counter(self.n_grams)
        return self._counts


@dataclass
class NGramProbability:
    """
    Probability model for n-grams
    Args:
        data List[List[str]]: The source for the n-grams
        n int: Size of the N-Gram
        k float: Smoothing parameter. Positive constant
        augment_vocabulary bool: Hack because the two probability functions use different vocabularies
        end_token str: End of sentence token
        unknown_token str: Unknown token
    """
    data: List[List[str]]
    n: int
    k: float = 1.0
    augment_vocabulary: bool = True
    end_token: str = "<e>"
    unknown_token: str = "<unk>"
    _n_grams: Optional[NGrams] = None
    _n_plus1_grams: Optional[NGrams] = None
    _vocabulary: Optional[FrozenSet] = None
    _vocabulary_size: Optional[int] = 0
    _probabilities: Optional[Dict[str, float]] = field(default_factory=dict)

    @property
    def n_grams(self) -> NGrams:
        if not self._n_grams:
            self._n_grams = NGrams(data=self.data, n=self.n)
        return self._n_grams

    @property
    def n_plus1_grams(self) -> NGrams:
        if not self._n_plus1_grams:
            self._n_plus1_grams = NGrams(data=self.data, n=self.n+1)
        return self._n_plus1_grams

    @property
    def vocabulary(self) -> FrozenSet:
        """Unique words in the dictionary"""
        if not self._vocabulary:
            data = list(chain.from_iterable(self.data)).copy()
            if self.augment_vocabulary:
                data.extend([self.end_token, self.unknown_token])
            self._vocabulary = frozenset(data)
        return self._vocabulary

    @property
    def vocabulary_size(self) -> int:
        """Number of unique tokens in the data"""
        if not self._vocabulary_size:
            self._vocabulary_size = len(self.vocabulary)
        return self._vocabulary_size

    def probability(self, word: str, previous_n_gram: Tuple[str]) -> float:
        """
        Calculates the probabiltiy of the word, given the previous n-gram
            Args: 
                word str: next probable word after `previous_n_gram`
                previous_n_gram Tuple[str]: Sequence of words of length `n`
            Returns:
                probability float: probability of the word after `previous_n_gram`
        """

        previous_n_gram = tuple(previous_n_gram)
        previous_n_gram_count = self.n_grams.counts.get(previous_n_gram, 0)
        denominator = previous_n_gram_count + self.k * self.vocabulary_size

        n_plus1_gram = previous_n_gram + (word,)
        n_plus1_gram_count = self.n_plus1_grams.counts.get(n_plus1_gram, 0)
        numerator = n_plus1_gram_count + self.k

        probability = numerator/denominator
        return probability

    def probabilities(self, previous_n_gram: Tuple[str]) -> Dict[str, float]:
        """
        Finds the probability of each word in the vocabulary
        Args:
            previous_n_gram Tuple[str]: Sequence of words of length `n`
        Returns:
            {word: <probability of word following `previous_n_gram` for the vocabulary}
        """
        return {word: self.probability(word=word, previous_n_gram=previous_n_gram) for word in self.vocabulary}


@dataclass
class Perplexity:
    """
    Calculate perplexity
        Args:
            data List[List[str]]: Tokenized training input
            n int: Size of the n-grams
            augment_vocabulary bool: Whether to augment the vocabulary for toy examples 
    """
    data: List[List[str]]
    n: int
    augment_vocabulary: bool
    _probablifier: Optional[NGramProbability] = None

    @property
    def probabilifier(self):
        """Probability Calculator"""
        if not self._probablifier:
            self._probablifier = NGramProbability(data=self.data,
                                                  n=self.n,
                                                  augment_vocabulary=self.augment_vocabulary)
        return self._probablifier

    def perplexity(self, sentence: List[str]) -> float:
        """Calculates the perplexity of a sentence"""
        sentence = tuple(["<s>"]*self.n + sentence + ["<e>"])
        N = len(sentence)

        n_grams = (sentence[position-self.n:position] for position in range(self.n, N))

        words = (sentence[position] for position in range(self.n, N))

        words_n_grams = zip(words, n_grams)
        probabilities = (self.probabilifier.probability(word, n_gram) for word, n_gram in words_n_grams)

        product = math.prod((1/probability for probability in probabilities))
        return product**(1/N)


@dataclass
class WordSuggester:
    # TODO: Add a docstring for this class
    """

    """
    data: List[List[str]]
    n: int
    _probablifier: Optional[NGramProbability] = None

    @property
    def probablifier(self):
        """Probability Calculator"""
        if not self._probablifier:
            self._probablifier = NGramProbability(data=self.data,
                                                  n=self.n)
        return self._probablifier

    def suggest_a_word(self,
                       previous_tokens: List[str],
                       start_with: Optional[str] = None):
        """
        Get suggestion for the next word
        Args:
            previous_tokens List[str]: Input sentence where each token is a word. 
            Sentence must have length > n
            start_with: If not None, specifies the first few letters of the next word
        Returns:
        """

        # Get the most recent `n` words as the previous n-gram from the words
        # that the user already typed
        previous_n_gram = previous_tokens[-self.n:]

        # Estimate the probabilities that each word in the vocabulary is the next word
        probabilities = self.probablifier.probabilities(previous_n_gram=previous_n_gram)

        # Suggestion will be set to the word with highest probability
        suggestion = None
        max_probability = 0

        for word, probability in probabilities.items():
            if start_with is not None and not word.startswith(start_with):
                continue

            # Check if this word's probability is greater than the current maximum probability
            if probability > max_probability:
                suggestion = word
                max_probability = probability

        return suggestion, max_probability


def get_suggestions(sentences: List[List[str]],
                    previous_tokens: List[str],
                    model_counts: int = 5,
                    start_with: Optional[str] = None) -> Set[Tuple[str, float]]:

    models = [WordSuggester(data=sentences, n=i) for i in range(1, model_counts+1)]
    suggestions = set()

    for i in range(model_counts):
        suggestion = models[i].suggest_a_word(previous_tokens=previous_tokens, start_with=start_with)
        suggestions.add(suggestion)

    return suggestions


if __name__ == "__main__":
    # tokenizer = Tokenizer(source=df["instruction_output"].head())
    # print(len(tokenizer.sentences))
    # splitter = TrainTestSplit(data=tokenizer.sentences)
    # print(splitter.shuffled)
    # print(splitter.split)
    # print(len(splitter.training))
    # print(len(splitter.testing))
    # processor = CountProcessor(training=splitter.training, testing=splitter.testing)
    # print(processor.vocabulary)
    # print(processor.closed_vocabulary)
    # print(processor.training_data_unknowns)
    # print(processor.testing_data_unknowns)

    # sentences = [["i", "like", "a", "cat"],
    #              ["this", "dog", "is", "like", "a", "cat"]]
    # # *** Unigram ***
    # expected = {('<s>',): 2, ('i',): 1, ('like',): 2, ('a',): 2, ('cat',): 2,
    #             ('<e>',): 2, ('this',): 1, ('dog',): 1, ('is',): 1}

    # uni_grams = NGrams(data=sentences, n=1)
    # print(uni_grams.n_grams)
    # print(uni_grams.counts)
    # expect(uni_grams.counts).to(have_keys(expected))

    # # *** Bigram ***
    # expected = {('<s>', '<s>'): 2, ('<s>', 'i'): 1, ('i', 'like'): 1,
    #             ('like', 'a'): 2, ('a', 'cat'): 2, ('cat', '<e>'): 2,
    #             ('<s>', 'this'): 1, ('this', 'dog'): 1, ('dog', 'is'): 1,
    #             ('is', 'like'): 1}
    # bi_grams = NGrams(data=sentences, n=2)
    # print(bi_grams.n_grams)
    # print(bi_grams.counts)
    # expect(bi_grams.counts).to(have_keys(expected))

    # model = NGramProbability(data=sentences, n=1, augment_vocabulary=False)

    # actual = model.probability("cat", ("a",))
    # expected = 0.3333
    # print(f"The estimated probability of word 'cat' given previous n-gram 'a' is {actual:.4f}")
    # expect(math.isclose(actual, expected, abs_tol=1e-4)).to(be_true)

    # # Probabilities test examples assuming you did augment the vocabulary
    # model = NGramProbability(data=sentences, n=1)
    # actual = model.probabilities(("a",))
    # expected = {'cat': 0.2727272727272727, 'i': 0.09090909090909091, 'like': 0.09090909090909091,
    #             'dog': 0.09090909090909091, 'is': 0.09090909090909091, 'this': 0.09090909090909091,
    #             '<unk>': 0.09090909090909091, 'a': 0.09090909090909091, '<e>': 0.09090909090909091}
    # print(actual)
    # expect(actual).to(have_keys(expected))

    # model = NGramProbability(data=sentences, n=2)
    # actual = model.probabilities(("<s>", "<s>"))
    # expected = {'this': 0.18181818181818182, 'like': 0.09090909090909091, '<unk>': 0.09090909090909091,
    #             'a': 0.09090909090909091, 'dog': 0.09090909090909091, 'cat': 0.09090909090909091,
    #             'i': 0.18181818181818182, 'is': 0.09090909090909091, '<e>': 0.09090909090909091}
    # print(actual)
    # expect(actual).to(have_keys(expected))

    # sentences = [["i", "like", "a", "cat"],
    #              ["this", "dog", "is", "like", "a", "cat"]]

    # model = Perplexity(data=sentences, n=1, augment_vocabulary=False)

    # expected = 2.8040
    # actual = model.perplexity(sentence=sentences[0])
    # print(f"Perplexity for the first train sample: {actual:.4f}")
    # expect(math.isclose(actual, expected, abs_tol=1e-4)).to(be_true)

    # test_sentence = ["i", "like", "a", "dog"]
    # expected = 3.9654
    # actual = model.perplexity(sentence=test_sentence)
    # print(f"Perplexity for the test sample: {actual:.4f}")
    # expect(math.isclose(actual, expected, abs_tol=1e-4)).to(be_true)

    # # Suggest a word
    # word_suggestor = WordSuggester(data=sentences, n=1)
    # previous_tokens = ["i", "like"]
    # suggestion, probability = word_suggestor.suggest_a_word(previous_tokens=previous_tokens)

    # print(f"The previous words are {previous_tokens} and the suggested word is {suggestion} "
    #       f"with a probability of {probability:.4f}")
    # expected_word, expected_probability = "a", 0.2727
    # expect(suggestion).to(equal(expected_word))
    # expect(math.isclose(probability, expected_probability, abs_tol=1e-4)).to(be_true)

    # # Test when setting the start_with
    # tmp_starts_with = "c"
    # suggestion, probability = word_suggestor.suggest_a_word(previous_tokens=previous_tokens, start_with=tmp_starts_with)
    # print(f"The previous words are {previous_tokens}, the suggestion must start with {tmp_starts_with} "
    #       f"and the suggested word is {suggestion} with a probability of {probability:.4f}")
    # expected_word, expected_probability = "cat", 0.0909
    # expect(suggestion).to(equal(expected_word))
    # expect(math.isclose(probability, expected_probability, abs_tol=1e-4)).to(be_true)

    # # Multiple suggestions
    # suggestions = get_suggestions(sentences=sentences, previous_tokens=previous_tokens)
    # print(f"The previous words are {previous_tokens}, the suggestions are:")
    # print(suggestions)

    # Multiple Word Suggestions
    tokenizer = Tokenizer(source=df["instruction_output"])
    print("tokenizer sentences")
    print(tokenizer.sentences[:20])
    splitter = TrainTestSplit(data=tokenizer.sentences)
    processor = CountProcessor(training=splitter.training, testing=splitter.testing)
    processed_train_data = processor.training_data_unknowns
    print("processed_train_data")
    print(processed_train_data[:20])

    print(get_suggestions(sentences=processed_train_data,
          previous_tokens=['in', 'this', 'post', 'we', 'will', 'discuss', 'two', 'n-gram', 'based', 'approaches'],
                          model_counts=5))
