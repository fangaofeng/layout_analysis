import importlib
import logging
from collections import defaultdict
from pathlib import Path
from string import whitespace

import pkg_resources

try:
    import spacy
    from spacy.cli import download
    from spacy import util
    from spacy.tokens import Doc
except Exception:
    raise Exception("spaCy not installed. Use `pip install spacy`.")


class Spacy(object):
    """
    spaCy
    https://spacy.io/

    Models for each target language needs to be downloaded using the
    following command:

    python -m spacy download en

    Default named entity types

    PERSON      People, including fictional.
    NORP        Nationalities or religious or political groups.
    FACILITY    Buildings, airports, highways, bridges, etc.
    ORG         Companies, agencies, institutions, etc.
    GPE         Countries, cities, states.
    LOC         Non-GPE locations, mountain ranges, bodies of water.
    PRODUCT     Objects, vehicles, foods, etc. (Not services.)
    EVENT       Named hurricanes, battles, wars, sports events, etc.
    WORK_OF_ART Titles of books, songs, etc.
    LANGUAGE    Any named language.

    DATE        Absolute or relative dates or periods.
    TIME        Times smaller than a day.
    PERCENT     Percentage, including "%".
    MONEY       Monetary values, including unit.
    QUANTITY    Measurements, as of weight or distance.
    ORDINAL     "first", "second", etc.
    CARDINAL    Numerals that do not fall under another type.

    """

    def __init__(self, lang):
        self.logger = logging.getLogger(__name__)
        self.name = "spacy"
        self.languages = ["vi","en", "de", "es", "pt", "fr", "it", "nl", "xx"]
        self.alpha_languages = {"ja": "Japanese", "zh": "Chinese"}

        self.lang = lang
        self.model = None

        # self.model = self.load_lang_model()

    def has_tokenizer_support(self):
        return self.lang and (
            self.has_NLP_support() or self.lang in self.alpha_languages
        )

    def has_NLP_support(self):
        return self.lang and (self.lang in self.languages)

    @staticmethod
    def is_package(name):
        """Check if string maps to a package installed via pip.

        name (unicode): Name of package.
        RETURNS (bool): True if installed package, False if not.

        From https://github.com/explosion/spaCy/blob/master/spacy/util.py
        """
        name = name.lower()  # compare package name against lowercase name
        packages = pkg_resources.working_set.by_key.keys()
        for package in packages:
            if package.lower().replace("-", "_") == name:
                return True
        return False

    @staticmethod
    def model_installed(name):
        """Check if spaCy language model is installed.

        From https://github.com/explosion/spaCy/blob/master/spacy/util.py

        :param name:
        :return:
        """
        data_path = util.get_data_path()
        if not data_path or not data_path.exists():
            raise IOError(f"Can't find spaCy data path: {data_path}")
        if name in {d.name for d in data_path.iterdir()}:
            return True
        if Spacy.is_package(name):  # installed as package
            return True
        if Path(name).exists():  # path to model data directory
            return True
        return False

    def load_lang_model(self):
        """
        Load spaCy language model or download if model is available and not
        installed.

        Currenty supported spaCy languages

        en English (50MB)
        de German (645MB)
        fr French (1.33GB)
        es Spanish (377MB)

        :return:
        """
        if self.lang == 'vi':
            model = spacy.load('vi_spacy_model')
        elif self.lang in self.languages:
            if not Spacy.model_installed(self.lang):
                download(self.lang)
            model = spacy.load(self.lang)
        elif self.lang in self.alpha_languages:
            language_module = importlib.import_module(f"spacy.lang.{self.lang}")
            language_method = getattr(language_module, self.alpha_languages[self.lang])
            model = language_method()
        self.model = model

    def enrich_sentences_with_NLP(self, all_sentences):
        """
        Enrich a list of fonduer Sentence objects with NLP features. We merge
        and process the text of all Sentences for higher efficiency.

        :param all_sentences: List of fonduer Sentence objects for one document
        :return:
        """
        if not self.has_NLP_support():
            raise NotImplementedError(
                f"Language {self.lang} not available in spacy beyond tokenization"
            )

        if len(all_sentences) == 0:
            return  # Nothing to parse

        if self.model.has_pipe("sentencizer"):
            self.model.remove_pipe("sentencizer")
            self.logger.debug(
                f"Removed sentencizer ('sentencizer') from model. "
                f"Now in pipeline: {self.model.pipe_names}"
            )

        if self.model.has_pipe("sentence_boundary_detector"):
            self.model.remove_pipe(name="sentence_boundary_detector")
        self.model.add_pipe(
            set_custom_boundary, before="parser", name="sentence_boundary_detector"
        )

        sentence_batches = self._split_sentences_by_char_limit(
            all_sentences, self.model.max_length
        )

        # TODO: We could do this in parallel. Test speedup in the future
        for sentence_batch in sentence_batches:
            custom_tokenizer = TokenPreservingTokenizer(self.model.vocab)
            # we circumvent redundant tokenization by using a custom
            # tokenizer that directly uses the already separated words
            # of each sentence as tokens
            doc = custom_tokenizer(sentence_batch)
            doc.user_data = sentence_batch
            for name, proc in self.model.pipeline:  # iterate over components in order
                doc = proc(doc)

            try:
                assert doc.is_parsed
            except Exception:
                self.logger.exception(f"{doc} was not parsed")

            for sent, current_sentence_obj in zip(doc.sents, sentence_batch):
                parts = defaultdict(list)

                for i, token in enumerate(sent):
                    parts["lemmas"].append(token.lemma_)
                    parts["pos_tags"].append(token.tag_)
                    parts["ner_tags"].append(
                        token.ent_type_ if token.ent_type_ else "O"
                    )
                    head_idx = (
                        0 if token.head is token else token.head.i - sent[0].i + 1
                    )
                    parts["dep_parents"].append(head_idx)
                    parts["dep_labels"].append(token.dep_)
                current_sentence_obj.pos_tags = parts["pos_tags"]
                current_sentence_obj.lemmas = parts["lemmas"]
                current_sentence_obj.ner_tags = parts["ner_tags"]
                current_sentence_obj.dep_parents = parts["dep_parents"]
                current_sentence_obj.dep_labels = parts["dep_labels"]
                yield current_sentence_obj

    def _split_sentences_by_char_limit(self, all_sentences, batch_char_limit):
        sentence_batches = [[]]
        num_chars = 0
        for sentence in all_sentences:
            if num_chars + len(sentence.text) >= batch_char_limit:
                sentence_batches.append([sentence])
                num_chars = len(sentence.text)
            else:
                sentence_batches[-1].append(sentence)
                num_chars += len(sentence.text)
        return sentence_batches

    def split_sentences(self, text):
        """
        Split input text into sentences that match CoreNLP's default format,
        but are not yet processed.

        :param text: The text of the parent paragraph of the sentences
        :return:
        """

        if self.model.has_pipe("sentence_boundary_detector"):
            self.model.remove_pipe(name="sentence_boundary_detector")

        if not self.model.has_pipe("sentencizer"):
            sentencizer = self.model.create_pipe("sentencizer")  # add sentencizer
            self.model.add_pipe(sentencizer)
        try:
            doc = self.model(text, disable=["parser", "tagger", "ner"])
        except ValueError:
            # temporary increase character limit of spacy
            # 'Probably save' according to spacy, as no parser or NER is used
            previous_max_length = self.model.max_length
            self.model.max_length = 100_000_000
            self.logger.warning(
                f"Temporarily increased spacy maximum "
                f"character limit to {self.model.max_length} to split sentences."
            )
            doc = self.model(text, disable=["parser", "tagger", "ner"])
            self.model.max_length = previous_max_length
            self.logger.warning(
                f"Spacy maximum "
                f"character limit set back to {self.model.max_length}."
            )

        doc.is_parsed = True
        position = 0
        for sent in doc.sents:
            parts = defaultdict(list)
            text = sent.text

            for i, token in enumerate(sent):
                parts["words"].append(str(token))
                parts["lemmas"].append(token.lemma_)
                parts["pos_tags"].append(token.pos_)
                parts["ner_tags"].append("")  # placeholder for later NLP parsing
                parts["char_offsets"].append(token.idx)
                parts["abs_char_offsets"].append(token.idx)
                parts["dep_parents"].append(0)  # placeholder for later NLP parsing
                parts["dep_labels"].append("")  # placeholder for later NLP parsing

            # make char_offsets relative to start of sentence
            parts["char_offsets"] = [
                p - parts["char_offsets"][0] for p in parts["char_offsets"]
            ]
            parts["position"] = position
            parts["text"] = text

            position += 1

            yield parts


def set_custom_boundary(doc):
    """Set the sentence boundaries based on the already separated sentences.
    :param doc: doc.user_data should have a list of Sentence.
    :return doc:
    """
    if doc.user_data == {}:
        raise AttributeError("A list of Sentence is not attached to doc.user_data.")
    # Set every token.is_sent_start False because they are all True by default
    for token_nr, token in enumerate(doc):
        doc[token_nr].is_sent_start = False
    # Set token.is_sent_start True when it is the first token of a Sentence
    token_nr = 0
    for sentence in doc.user_data:
        doc[token_nr].is_sent_start = True
        token_nr += len(sentence.words)
    return doc


class TokenPreservingTokenizer(object):
    """
    This custom tokenizer simply preserves the tokenization that was already
    performed during sentence splitting. It will output a list of space
    separated tokens, whereas each token is a single word from the list of
    sentences.
    """

    def __init__(self, vocab):
        """Initialize a custom tokenizer.

        :param vocab: The vocab attribute of the respective spacy language object.
        """
        self.logger = logging.getLogger(__name__)
        self.vocab = vocab

    def __call__(self, tokenized_sentences):
        """Apply the custom tokenizer.

        :param tokenized_sentences: A list of sentences that was previously
        tokenized/split by spacy
        :return: Doc (a container for accessing linguistic annotations).
        """
        all_input_tokens = []
        all_spaces = []
        for sentence in tokenized_sentences:
            words_in_sentence = sentence.words
            if len(words_in_sentence) > 0:
                all_input_tokens += sentence.words
                current_sentence_pos = 0
                spaces_list = [True] * len(words_in_sentence)
                # Last word in sentence always assumed to be followed by space
                for i, word in enumerate(words_in_sentence[:-1]):
                    current_sentence_pos = sentence.text.find(
                        word, current_sentence_pos
                    )
                    if current_sentence_pos == -1:
                        raise AttributeError(
                            "Could not find token in its parent sentence"
                        )
                    current_sentence_pos += len(word)
                    if not any(
                        sentence.text[current_sentence_pos:].startswith(s)
                        for s in whitespace
                    ):
                        spaces_list[i] = False
                all_spaces += spaces_list
        return Doc(self.vocab, words=all_input_tokens, spaces=all_spaces)
