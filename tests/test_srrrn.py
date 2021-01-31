import random
import unittest

import torch
from torch import nn

from model import SRRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
atol = 1e-04


class TestSRRRN(unittest.TestCase):

    def test_pack_padded_sequences(self):
        with torch.no_grad():
            for _ in range(1024):

                batch_size = random.randint(1, 15)
                seq_length = random.randint(1, 15)
                input_size = random.randint(1, 5)
                hidden_size = random.randint(1, 5)
                bias = bool(random.getrandbits(1))
                nonlinearity = random.choice(['tanh', 'relu'])
                mode = random.choice(['rnn', 'gru'])
                number_of_states = random.randint(1, 15)
                temperature = random.choice([1.00, 0.5, 0.1])

                seq = torch.rand(batch_size, seq_length, input_size).to(device)
                h_0 = torch.rand(batch_size, hidden_size).to(device)

                lengths = torch.randint(1, seq_length + 1, (batch_size,)).to('cpu')
                packed_sequence = nn.utils.rnn.pack_padded_sequence(
                    seq, lengths, batch_first=True, enforce_sorted=False
                )

                # without stochastic component
                rnn = SRRNN(
                    input_size, hidden_size, bias=bias, mode=mode, nonlinearity=nonlinearity
                )
                rnn.to(device)

                output, h_n, _ = rnn(seq, h_0)

                self.assertEqual(h_n.size(), (batch_size, hidden_size))
                self.assertEqual(output.size(), (batch_size, seq_length, hidden_size))

                output_packed, h_n_packed, _ = rnn(packed_sequence, h_0)

                self.assertEqual(h_n_packed.size(), (batch_size, hidden_size))
                self.assertEqual(output_packed.data.size(), (lengths.sum().item(), hidden_size))

                for i in range(batch_size):
                    self.assertTrue(torch.allclose(h_n_packed[i, :], output[i, lengths[i].item() - 1, :], atol=atol))

                # with stochastic component
                rnn = SRRNN(
                    input_size, hidden_size, bias=bias, mode=mode, nonlinearity=nonlinearity,
                    number_of_states=number_of_states, temperature=temperature
                )
                rnn.to(device)

                output, h_n, transition_probabilities = rnn(seq, h_0)

                self.assertEqual(h_n.size(), (batch_size, hidden_size))
                self.assertEqual(output.size(), (batch_size, seq_length, hidden_size))
                self.assertEqual(transition_probabilities.size(), (batch_size, seq_length, number_of_states))

                output_packed, h_n_packed, transition_probabilities_packed = rnn(packed_sequence, h_0)

                self.assertEqual(h_n_packed.size(), (batch_size, hidden_size))
                self.assertEqual(output_packed.data.size(), (lengths.sum().item(), hidden_size))
                self.assertEqual(transition_probabilities_packed.data.size(), (lengths.sum().item(), number_of_states))

                for i in range(batch_size):
                    self.assertTrue(torch.allclose(h_n_packed[i, :], output[i, lengths[i].item() - 1, :], atol=atol))

    def test_rnn_mode_compare_with_naive_implementation(self):
        with torch.no_grad():
            for _ in range(1024):

                batch_size = random.randint(1, 15)
                seq_length = random.randint(1, 15)
                input_size = random.randint(1, 5)
                hidden_size = random.randint(1, 5)
                bias = bool(random.getrandbits(1))
                nonlinearity = random.choice(['tanh', 'relu'])
                number_of_states = random.randint(1, 15)
                temperature = random.choice([1.00, 0.5, 0.1])

                seq = torch.rand(batch_size, seq_length, input_size).to(device)
                h_0 = torch.rand(batch_size, hidden_size).to(device)

                lengths = torch.randint(1, seq_length + 1, (batch_size,)).to('cpu')
                packed_sequence = nn.utils.rnn.pack_padded_sequence(
                    seq, lengths, batch_first=True, enforce_sorted=False
                )

                # without stochastic component
                rnn = SRRNN(
                    input_size, hidden_size, bias=bias, mode='rnn', nonlinearity=nonlinearity
                )
                rnn.to(device)

                output, h_n, _ = rnn(packed_sequence, h_0)
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

                Wh = rnn.rnn_cell.weight_hh.data.detach().clone()
                Wi = rnn.rnn_cell.weight_ih.data.detach().clone()
                if bias:
                    bh = rnn.rnn_cell.bias_hh.detach().clone()
                    bi = rnn.rnn_cell.bias_ih.detach().clone()

                for i in range(batch_size):
                    hi = h_0[i]
                    for j in range(lengths[i]):
                        hi = Wh @ hi + Wi @ seq[i, j, :]
                        if bias:
                            hi = hi + bi + bh
                        hi = torch.relu(hi) if nonlinearity == 'relu' else torch.tanh(hi)

                        self.assertTrue(torch.allclose(hi, output[i, j], atol=atol))
                        if j == lengths[i] - 1:
                            self.assertTrue(torch.allclose(hi, h_n[i], atol=atol))

                # with stochastic component
                rnn = SRRNN(
                    input_size, hidden_size, bias=bias, mode='rnn', nonlinearity=nonlinearity,
                    number_of_states=number_of_states, temperature=temperature
                )
                rnn.to(device)

                output, h_n, transition_probabilities = rnn(packed_sequence, h_0)
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                transition_probabilities, _ = nn.utils.rnn.pad_packed_sequence(
                    transition_probabilities, batch_first=True
                )

                Wh = rnn.rnn_cell.weight_hh.data.detach().clone()
                Wi = rnn.rnn_cell.weight_ih.data.detach().clone()
                if bias:
                    bh = rnn.rnn_cell.bias_hh.detach().clone()
                    bi = rnn.rnn_cell.bias_ih.detach().clone()
                states = rnn.states.data.detach().clone()

                for i in range(batch_size):
                    hi = h_0[i]
                    for j in range(lengths[i]):
                        hi_ = Wh @ hi + Wi @ seq[i, j, :]
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

                        self.assertTrue(torch.allclose(hi, output[i, j], atol=atol))
                        if j == lengths[i] - 1:
                            self.assertTrue(torch.allclose(hi, h_n[i], atol=atol))


if __name__ == '__main__':
    unittest.main()
