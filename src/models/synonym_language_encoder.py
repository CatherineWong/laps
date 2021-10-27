"""
synonym_language_encoder.py | Author: Catherine Wong.

Implements utilities for generating stochastic synonym replacement in language, useful for data augmentation and for testing 'naturalistic' synthetic language.
"""
import random
from src.task_loaders import *
import src.utils as utils
import src.models.model_loaders as model_loaders

LanguageEncoderRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.LANGUAGE_ENCODER
]


@LanguageEncoderRegistry.register
class SynonymLanguageEncoder(model_loaders.ModelLoader):
    """
    Language 'encoder' that replaces language with assignments under a synonym lexicon.
    Uses uniform replacement for synonyms.
    """

    name = "synonym_language_encoder"

    WORD_TOKENIZE = "word_tokenize"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return SynonymLanguageEncoder(experiment_state=experiment_state, **kwargs)

    def __init__(
        self,
        experiment_state=None,
        tokenizer_fn=WORD_TOKENIZE,
    ):
        super().__init__()

        self.tokenizer_fn, self.tokenizer_cache = self._init_tokenizer(tokenizer_fn)

    def _init_tokenizer(self, tokenizer_fn):
        tokenizer_cache = dict()
        if tokenizer_fn == self.WORD_TOKENIZE:
            from nltk.tokenize import word_tokenize

            return word_tokenize, tokenizer_cache
        else:
            assert False

    def generate_synomym_language(
        self,
        experiment_state,
        task_split=TRAIN,
        task_batch_ids=ALL,
        synonym_file_path=None,
        keep_original=True,
        max_samples_per_synonyms=True,
    ):
        """
        Replaces the language in tasks with language based on synonyms.
        """
        # Load language for tasks.
        tasks_to_language = experiment_state.get_language_and_tasks_for_ids(
            task_split,
            task_batch_ids,
        )
        # Load synonym dict.
        with open(synonym_file_path) as f:
            word2synonyms = json.load(f)
        # Generate synonyms
        tasks_to_synonym_language = {}
        for task_id in tasks_to_language:
            for sentence in tasks_to_language[task_id]:
                tasks_to_synonym_language[
                    task_id
                ] = self.generate_synonym_language_for_sentence(
                    sentence,
                    word2synonyms,
                    keep_original=keep_original,
                    max_samples_per_synonyms=max_samples_per_synonyms,
                )
        # Set all of the tasks to this.
        for task_id in tasks_to_synonym_language:
            experiment_state.task_language[task_split][
                task_id
            ] = tasks_to_synonym_language[task_id]
        return tasks_to_synonym_language

    def generate_synonym_language_for_sentence(
        self,
        sentence,
        word2synonyms,
        keep_original=True,
        max_samples_per_synonyms=True,
    ):
        """
        :ret: [array of synonym sentences]
        """
        synonym_sentences = set() if not keep_original else set([sentence])
        sentence_tokens = self.tokenizer_fn(sentence)

        max_tries = max_samples_per_synonyms * 2

        curr_tries = 0
        while len(synonym_sentences) < max_samples_per_synonyms:
            if curr_tries >= max_tries:
                break
            synonym_sentence = []
            for token in sentence_tokens:
                if token in word2synonyms:
                    synonym_token = random.choice(word2synonyms[token] + [token])
                else:
                    synonym_token = token
                synonym_sentence.append(synonym_token)
            synonym_sentence = " ".join(synonym_sentence)
            if synonym_sentence == sentence and not keep_original:
                pass
            else:
                synonym_sentences.add(synonym_sentence)
        return list(synonym_sentences)
