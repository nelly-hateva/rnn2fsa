import torch
from torch import nn


class SRRNN(nn.Module):
    r"""
    https://arxiv.org/pdf/1901.08817.pdf

    https://github.com/deepsemantic/sr-rnns

    Applies a single-layer State Regularized RNN to an input sequence.

    If :attr:`mode` is ``'rnn'``, then for each element in the input sequence computes the function

      .. math::
        h_t' = \text{relu}(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

      where :math:`x_t` is the input at time `t`, :math:`h_{(t-1)}` is the hidden state
      at time `t-1` or the initial hidden state at time `0`.
      If :attr:`nonlinearity` is ``'tanh'``, then `tanh` is used instead of `relu`.

    If :attr:`mode` is ``'gru'``, then for each element in the input sequence computes the function

      .. math::
        r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr})
      .. math::
          z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz})
      .. math::
          n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)} + b_{hn}))
      .. math::
          h_t' = (1 - z_t) * n_t + z_t * h_{(t-1)}

      where :math:`x_t` is the input at time `t`, :math:`h_{(t-1)}` is the hidden state
      at time `t-1` or the initial hidden state at time `0`.
      :math:`r_t`, :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
      :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    In both modes, if :attr:`number_of_states` is ``None``, then the hidden state
    at time `t` :math:`h_t` equals :math:`h_t'`.

    Otherwise,

      .. math::
        \alpha_{i} = \frac{\exp(- \Vert{h_t' - s_i}\Vert / \tau)}{\sum_{j=1}^{k} \exp(- \Vert{h_t' - s_j}\Vert / \tau)}
      .. math::
        h_t = {\sum_{i=1}^{k} \alpha_{i} s_i}

      where :math:`\{s_1, s_2, ..., s_k\}` are the k learnable states.
      :math:`\alpha_{i}` is the probability of the RNN to transition to state i
      given the vector :math:`h_t'` for which we write :math:`p_{h_t'}(i) = \alpha_{i}`
      :math:`\tau` is a temperature parameter that can be used to anneal
      the probabilistic state transition behavior. The lower :math:`\tau` the
      more :math:`\alpha` resembles the one-hot encoding of a centroid. The
      higher :math:`\tau` the more uniform is :math:`\alpha`

    Args:
        - input_size: The number of expected features in the input `x`
        - hidden_size: The number of features in the hidden state `h`
        - mode: The RNN mode to use.
          Can be either ``'rnn'`` or ``'gru'``. Default: ``'rnn'``
        - nonlinearity: The non-linearity to use if :attr:`mode` is ``'rnn'``.
          Can be either ``'tanh'`` or ``'relu'``. Default: ``'relu'``
        - bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        - number_of_states: The number of learnable finite states.
          If ``None`` then the stochastic component is not used. Default: ``None``
        - temperature: The temperature parameter. Default: ``1.00``

    Inputs: input, h_0
        - **input** of shape `(batch, seq_len, input_size)`:
          tensor containing the features of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If `h_0` is not provided, it defaults to zero.

    Outputs: output, h_n
        - **output** of shape `(batch, seq_len, hidden_size)`: tensor
          containing the output features `(h_t)` from the RNN for each `t`.
          If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** of shape `(batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

    Attributes:
        If :attr:`mode` is ``'rnn'``:
            - weight_ih: the learnable input-hidden weights of shape `(hidden_size, input_size)`
            - weight_hh: the learnable hidden-hidden weights of shape `(hidden_size, hidden_size)`
            - bias_ih: the learnable input-hidden bias of shape `(hidden_size)`
            - bias_hh: the learnable hidden-hidden bias of shape `(hidden_size)`
        If :attr:`mode` is ``'gru'``:
            - weight_ih : the learnable input-hidden weights (W_ir|W_iz|W_in) of shape `(3*hidden_size, input_size)`
            - weight_hh : the learnable hidden-hidden weights (W_hr|W_hz|W_hn) of shape `(3*hidden_size, hidden_size)`
            - bias_ih : the learnable input-hidden bias (b_ir|b_iz|b_in) of shape `(3*hidden_size)`
            - bias_hh : the learnable hidden-hidden bias(b_hr|b_hz|b_hn) of shape `(3*hidden_size)`
        If :attr:`number_of_states` is not ``None``:
            - states: the learnable finite number of states of shape `(number_of_states, hidden_size)

    Examples:
        >>> rnn = SRRNN(10, 20)
        >>> seq = torch.randn(6, 3, 10)
        >>> h_0 = torch.randn(6, 20)
        >>> output, h_n = rnn(seq, h_0)
    """

    def __init__(
            self, input_size, hidden_size, bias=True, mode='rnn', nonlinearity='relu',
            number_of_states=None, temperature=1.00
    ):
        super(SRRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        if mode == 'rnn':
            self.rnn_cell = nn.RNNCell(
                input_size, hidden_size, bias=bias, nonlinearity=nonlinearity
            )
        elif mode == 'gru':
            self.rnn_cell = nn.GRUCell(
                input_size, hidden_size, bias=bias
            )
        else:
            raise ValueError("Unknown mode '{}'".format(mode))

        self.stochastic_component = False

        if number_of_states:
            self.stochastic_component = True
            self.number_of_states = number_of_states

            self.softmax = nn.Softmax(dim=1)
            self.states = nn.Parameter(
                torch.Tensor(self.number_of_states, hidden_size)
            )
            self.temperature = temperature

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.eye_(self.rnn_cell.weight_hh)
        # nn.init.kaiming_normal_(self.rnn_cell.weight_ih, nonlinearity=self.nonlinearity)
        # if self.rnn_cell.bias:
        #   nn.init.zeros_(self.rnn_cell.bias_hh)
        #   nn.init.zeros_(self.rnn_cell.bias_ih)

        if self.stochastic_component:
            # nn.init.normal_(self.states, mean=0.0, std=1.0)
            nn.init.kaiming_normal_(self.states, nonlinearity=self.nonlinearity)
            # nn.init.kaiming_uniform_(self.states, nonlinearity=self.nonlinearity)

    def extra_repr(self):
        s = ''
        if 'number_of_states' in self.__dict__:
            s = 'number_of_states={number_of_states}'
            if 'temperature' in self.__dict__ and self.temperature != 1.00:
                s += ', temperature={temperature}'

        return s.format(**self.__dict__)

    @staticmethod
    def permute_hidden(hidden, permutation, dim=0):
        if permutation is None:
            return hidden
        return hidden.index_select(dim, permutation)

    def forward(self, input_, h_0=None):
        orig_input = input_

        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            input_, batch_sizes, sorted_indices, unsorted_indices = input_
            max_batch_size = int(batch_sizes[0])
        else:
            max_batch_size, sorted_indices = input_.size(0), None

        if h_0 is None:
            h_0 = torch.zeros(
                max_batch_size, self.hidden_size, dtype=input_.dtype, device=input_.device
            )
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            h_0 = self.permute_hidden(h_0, sorted_indices)

        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            output, hidden, transition_probabilities = self.forward_packed(input_, batch_sizes, h_0)

            if self.stochastic_component:
                transition_probabilities = nn.utils.rnn.PackedSequence(
                    transition_probabilities, batch_sizes, sorted_indices, unsorted_indices
                )

            hidden = self.permute_hidden(hidden, unsorted_indices)
            output = nn.utils.rnn.PackedSequence(
                output, batch_sizes, sorted_indices, unsorted_indices
            )

            return output, hidden, transition_probabilities

        return self.forward_tensor(input_, h_0)

    def forward_tensor(self, input_, h_0):
        output, h_t, transition_probabilities = [], h_0, []

        for t in range(input_.size(1)):
            result = self.forward_impl(input_[:, t, :], h_t)
            if self.stochastic_component:
                probabilities, h_t = result
                transition_probabilities.append(probabilities)
            else:
                h_t = result
            output.append(h_t)

        output = torch.stack(output).permute(1, 0, 2)

        if self.stochastic_component:
            return output, h_t, torch.stack(transition_probabilities).permute(1, 0, 2)

        return output, h_t, None

    def forward_packed(self, input_, batch_sizes, h_0):
        output, h_n, t, h_t, transition_probabilities = [], [], 0, h_0, []

        for batch_size in batch_sizes:
            batch_size = int(batch_size)

            h_t, h_n_ = h_t[:batch_size], h_t[batch_size:]
            h_n.append(h_n_)
            result = self.forward_impl(input_[t: t + batch_size], h_t)

            if self.stochastic_component:
                probabilities, h_t = result
                transition_probabilities.append(probabilities)
            else:
                h_t = result

            output.append(h_t)
            t += batch_size

        h_n.append(h_t)
        h_n.reverse()

        output, h_n = torch.cat(output), torch.cat(h_n)

        if self.stochastic_component:
            return output, h_n, torch.cat(transition_probabilities)

        return output, h_n, None

    def forward_impl(self, input_, h_t):
        h_t_ = self.rnn_cell(input_, h_t)
        if self.stochastic_component:
            transition_probabilities = self.softmax(
                (- torch.pow(self.states - h_t_.unsqueeze(1), 2).sum(2)) / self.temperature
            )
            return transition_probabilities, torch.matmul(transition_probabilities, self.states)
        return h_t_
