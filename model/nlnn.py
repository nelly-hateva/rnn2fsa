from torch import nn

from .srrrn import SRRNN


class NLNN(nn.Module):

    def __init__(self, params):
        super(NLNN, self).__init__()

        # + 1 because of the padding with 0s
        num_embeddings = params['num_embeddings'] + 1

        if 'embedding_dim' in params:
            embedding_dim = params['embedding_dim']
            self.embeddings = nn.Embedding(
                num_embeddings,
                embedding_dim,
                padding_idx=0
            )
            rnn_input_size = embedding_dim
        else:
            # one hot encoding
            self.embeddings = nn.Embedding(
                num_embeddings,
                num_embeddings,
                padding_idx=0
            )
            nn.init.eye_(self.embeddings.weight.data)
            self.embeddings.weight.requires_grad = False
            rnn_input_size = num_embeddings

        if 'number_of_states' in params:
            self.stochastic = True
            self.rnn = SRRNN(
                rnn_input_size, params['hidden_size'], bias=params['bias'],
                mode=params['mode'], nonlinearity=params['nonlinearity'],
                number_of_states=params['number_of_states'],
                temperature=params['temperature']
            )
        else:
            self.stochastic = False
            mode = params.pop('mode', 'rnn')
            if mode == 'rnn':
                self.rnn = nn.RNN(
                    rnn_input_size, params['hidden_size'], bias=params['bias'],
                    nonlinearity=params['nonlinearity'],
                    batch_first=True
                )
            elif mode == 'gru':
                self.rnn = nn.GRU(
                    rnn_input_size, params['hidden_size'], bias=params['bias'],
                    batch_first=True
                )
            else:
                raise ValueError("Unknown mode '{}'".format(mode))

        self.linear = nn.Linear(
            in_features=params['hidden_size'], out_features=2, bias=True
        )

    def forward(
            self, x, length, h_0=None, return_rnn_output=False, return_probabilities=False
    ):
        embedding_output = self.embeddings(x)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            embedding_output, length, batch_first=True, enforce_sorted=False
        )

        if self.stochastic:
            rnn_output, h_n, transition_probabilities = self.rnn(packed_sequence, h_0=h_0)
        else:
            rnn_output, h_n = self.rnn(packed_sequence, h_0)
            h_n = h_n[-1, :, :]

        linear_output = self.linear(h_n)

        if return_rnn_output and return_probabilities:
            return linear_output, rnn_output, transition_probabilities
        elif return_rnn_output:
            return linear_output, rnn_output
        elif return_probabilities:
            return linear_output, transition_probabilities
        else:
            return linear_output
