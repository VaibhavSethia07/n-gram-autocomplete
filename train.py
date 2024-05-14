import random
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

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
    training_fraction: float = 0.8
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
            self._training = self._shuffled[0:self._split]
        return self._training

    @property
    def testing(self):
        """Testing set"""
        if not self._testing:
            self._testing = self._shuffled[self._split:]
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
                if word not in self._closed_vocabulary_set:
                    replaced_sentence.append(self.unknown_token)
                else:
                    replaced_sentence.append(word)

            replaced_tokenized_sentences.append(replaced_sentence)

        return replaced_tokenized_sentences


if __name__ == "__main__":
    tokenizer = Tokenizer(source=df["instruction_output"].head())
    print(len(tokenizer.sentences))
    splitter = TrainTestSplit(data=tokenizer.sentences)
    print(splitter.shuffled)
    print(splitter.split)
    print(len(splitter.training))
    print(len(splitter.testing))
    processor = CountProcessor(training=splitter.training, testing=splitter.testing)
    print(processor.vocabulary)
    print(processor.closed_vocabulary)
    print(processor.training_data_unknowns)
    print(processor.testing_data_unknowns)
