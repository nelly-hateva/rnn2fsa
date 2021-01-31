import random
import unittest

import torch
from torch import nn

from model import NLNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
atol = 1e-04


class TestNLNN(unittest.TestCase):
    def test_rnn_mode_compare_with_naive_implementation(self):
        with torch.no_grad():
            for _ in range(1024):

                batch_size = random.randint(1, 15)
                seq_length = random.randint(1, 15)
                num_embeddings = random.randint(2, 5)
                embedding_dim = random.randint(3, 7)
                hidden_size = random.randint(1, 5)
                bias = bool(random.getrandbits(1))
                nonlinearity = random.choice(['tanh', 'relu'])
                number_of_states = random.randint(1, 15)
                temperature = random.choice([1.00, 0.5, 0.1])

                seq = torch.randint(1, num_embeddings + 1, (batch_size, seq_length)).long().to(device)
                h_0 = torch.rand(1, batch_size, hidden_size).to(device)
                lengths = torch.randint(1, seq_length + 1, (batch_size,)).to('cpu')

                # without stochastic component
                nlnn = NLNN(params={
                    'num_embeddings': num_embeddings,
                    'embedding_dim': embedding_dim,
                    'hidden_size': hidden_size,
                    'bias': bias,
                    'mode': 'rnn',
                    'nonlinearity': nonlinearity
                })
                nlnn.to(device)

                output, rnn_output = nlnn(seq, lengths, h_0, return_rnn_output=True)
                rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

                embedding = nlnn.embeddings.weight.data.detach().clone()
                Wh = nlnn.rnn.weight_hh_l0.data.detach().clone()
                Wi = nlnn.rnn.weight_ih_l0.data.detach().clone()
                if bias:
                    bh = nlnn.rnn.bias_hh_l0.detach().clone()
                    bi = nlnn.rnn.bias_ih_l0.detach().clone()
                A = nlnn.linear.weight.data.detach().clone()
                b = nlnn.linear.bias.data.detach().clone()

                for i in range(batch_size):
                    emb = torch.index_select(embedding, 0, seq[i])
                    hi = h_0[0, i]
                    for j in range(lengths[i]):
                        hi = Wh @ hi + Wi @ emb[j]
                        if bias:
                            hi = hi + bi + bh
                        hi = torch.relu(hi) if nonlinearity == 'relu' else torch.tanh(hi)
                        self.assertTrue(torch.allclose(hi, rnn_output[i, j], atol=atol))

                    linear_output = A @ hi + b
                    self.assertTrue(torch.allclose(linear_output, output[i], atol=atol))

                # with stochastic component
                nlnn = NLNN(params={
                    'num_embeddings': num_embeddings,
                    'embedding_dim': embedding_dim,
                    'hidden_size': hidden_size,
                    'bias': bias,
                    'mode': 'rnn',
                    'nonlinearity': nonlinearity,
                    'number_of_states': number_of_states,
                    'temperature': temperature
                })
                nlnn.to(device)

                output, rnn_output, probabilities = nlnn(
                    seq, lengths, h_0[0], return_rnn_output=True, return_probabilities=True
                )
                rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
                probabilities, _ = nn.utils.rnn.pad_packed_sequence(probabilities, batch_first=True)

                embedding = nlnn.embeddings.weight.data.detach().clone()
                Wh = nlnn.rnn.rnn_cell.weight_hh.data.detach().clone()
                Wi = nlnn.rnn.rnn_cell.weight_ih.data.detach().clone()
                if bias:
                    bh = nlnn.rnn.rnn_cell.bias_hh.detach().clone()
                    bi = nlnn.rnn.rnn_cell.bias_ih.detach().clone()
                A = nlnn.linear.weight.data.detach().clone()
                b = nlnn.linear.bias.data.detach().clone()
                states = nlnn.rnn.states.data.detach().clone()

                for i in range(batch_size):
                    emb = torch.index_select(embedding, 0, seq[i])
                    hi = h_0[0, i]
                    for j in range(lengths[i]):
                        hi_ = Wh @ hi + Wi @ emb[j]
                        if bias:
                            hi_ = hi_ + bi + bh
                        hi_ = torch.relu(hi_) if nonlinearity == 'relu' else torch.tanh(hi_)

                        distances = []
                        for k in range(states.size(0)):
                            distance = - ((hi_ - states[k]).pow(2).sum() / temperature)
                            distances.append(distance)
                        alpha = nn.Softmax(dim=0)(torch.tensor(distances))
                        hi = torch.zeros((hidden_size,))
                        for k in range(states.size(0)):
                            hi += (alpha[k] * states[k])
                        self.assertTrue(torch.allclose(hi, rnn_output[i, j], atol=atol))
                        self.assertTrue(torch.allclose(alpha, probabilities[i, j], atol=atol))

                    linear_output = A @ hi + b
                    self.assertTrue(torch.allclose(linear_output, output[i], atol=atol))


if __name__ == '__main__':
    unittest.main()
