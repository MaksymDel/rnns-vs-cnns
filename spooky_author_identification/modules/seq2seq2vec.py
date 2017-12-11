from overrides import overrides

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("seq2seq2vec")
class Seq2Seq2VecEncoder(Seq2VecEncoder):
    """
    Takes Seq2Seq encoder and Seq2Vec encoders and appyies them one after another
    Parameters
    ----------
    seq2seq_encoder : ``Seq2SeqEncoder``
        To get all encoder outputs
    seq2vec_encoder : ``Seq2VecEncoder``
        To achieve a single vector 
    """
    def __init__(self,
                 seq2seq_encoder: Seq2SeqEncoder,
                 seq2vec_encoder: Seq2VecEncoder) -> None:
        super(Seq2VecEncoder, self).__init__()
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder

    @overrides
    def get_input_dim(self) -> int:
        return seq2seq_encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.seq2vec_encoder.get_output_dim()

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:
        encoder_outputs = self.seq2seq_encoder(inputs, mask)
        encoded_sentence = self.seq2vec_encoder(encoder_outputs, mask)
        return encoded_sentence

    @classmethod
    def from_params(cls, params: Params) -> 'Seq2Seq2VecEncoder':
        seq2seq_encoder_params = params.pop("seq2seq_encoder")
        seq2vec_encoder_params = params.pop("seq2vec_encoder")
        seq2seq_encoder = Seq2SeqEncoder.from_params(seq2seq_encoder_params)
        seq2vec_encoder = Seq2VecEncoder.from_params(seq2vec_encoder_params)

        return cls(seq2seq_encoder=seq2seq_encoder,
                   seq2vec_encoder=seq2vec_encoder)