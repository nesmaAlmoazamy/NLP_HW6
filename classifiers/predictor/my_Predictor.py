from typing import *

import numpy as np
import torch
from allennlp.models import Model
from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('predictor')
class MyPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        inst =  self._dataset_reader.text_to_instance([str(t) for t in tokens])
        logits = self.predict_instance(inst)['logits']
        label_id = np.argmax(logits)
        return self._model.vocab.get_token_from_index(label_id, 'labels')
