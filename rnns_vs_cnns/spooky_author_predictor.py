from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('author_classifier')
class SpookyAuthorPredictor(Predictor):
    """
    Wrapper for any spooky author classifier model that takes in a sentence and returns
    a label for it.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = JustSpacesWordSplitter()

    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        # We're overriding `predict_json` directly, so we don't need this.  But I'd rather have a
        # useless stub here then make the base class throw a RuntimeError instead of a
        # NotImplementedError - the checking on the base class is worth it.
        raise RuntimeError("this should never be called")

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = inputs["sentence"]
        print(sentence)
        instance = self._dataset_reader.text_to_instance(sentence)

        output = self._model.forward_on_instance(instance, cuda_device)
        
        return sanitize(output)