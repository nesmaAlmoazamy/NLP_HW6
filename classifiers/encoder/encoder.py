from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.nn.util import get_final_encoder_states
import torch
from overrides import overrides

@Seq2VecEncoder.register('my-transformer')
class TransformerSeq2VecEncoder (Seq2VecEncoder):
	
	def __init__(self, input_dim, hidden_dim, num_layers, projection_dim, feedforward_hidden_dim, num_attention_heads):
		super(Seq2VecEncoder, self).__init__()
		self.input_dim = input_dim
		self.output_dim = hidden_dim
		self.encoder = StackedSelfAttentionEncoder(input_dim=input_dim,
		    hidden_dim=hidden_dim,
		    projection_dim=projection_dim,
		    feedforward_hidden_dim=feedforward_hidden_dim,
		    num_layers=num_layers,
		    num_attention_heads=num_attention_heads)
	
	@overrides
	def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
		output_seq = self.encoder(inputs, mask)
		output_vec = get_final_encoder_states(output_seq, mask)
		return output_vec
	def get_input_dim(self) -> int:
		return self.input_dim
	def get_output_dim(self) -> int:
		return self.output_dim
