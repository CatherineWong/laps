"""
list: grammar.py | Author: Sam Acquaviva.
Utility functions for loading in the DSLs for the CLEVR domain.
"""
from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.list.listPrimitives as listPrimitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]


@GrammarRegistry.register
class ListGrammarLoader(ModelLoader):
    """
    Loads the list domain grammar.
    Original source: dreamcoder/domains/list/listPrimitives.py
    """

    name = "list"  # Special handler for OCaml enumeration.
    
    # TODO: Should probably set as flag which set of primitives to use.
    # Could be: primitives, basePrimitives, bootstrapTarget, re2_list_v0, bootstrapTarget_extra, no_length, or
    # McCarthyPrimitives.
    def load_model(self, experiment_state):
        LIST_PRIMITIVES = listPrimitives.primitives()
        grammar = LAPSGrammar.uniform(LIST_PRIMITIVES)
        grammar.function_prefix = "list"
        return grammar
