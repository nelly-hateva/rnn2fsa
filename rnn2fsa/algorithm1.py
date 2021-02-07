from collections import defaultdict, Counter

import torch
from torch import nn

from automata import Automaton


class Algorithm1:
    @staticmethod
    def model_to_automaton(model, data_loader, inv_alphabet):
        initial_state, transitions = 0, set()

        for data in data_loader:
            x, length = data['x'], data['length']
            _, transition_probabilities = model(x, length, return_probabilities=True)
            transition_probabilities, _ = nn.utils.rnn.pad_packed_sequence(transition_probabilities, batch_first=True)

            for j in range(x.size(0)):
                prev_state = initial_state
                for k in range(length[j]):
                    symbol = inv_alphabet[x[j, k].item()]
                    next_state = torch.argmax(transition_probabilities[j, k]).item() + 1
                    transitions.add((prev_state, symbol, next_state))
                    prev_state = next_state

        hidden_size = model.rnn.rnn_cell.weight_hh.size(0)

        state_symbol_to_states = defaultdict(set)

        for transition in transitions:
            q1, a, q2 = transition
            state_symbol_to_states[(q1, a)].add(q2)

        count_non_deterministic, count_arbitrarily = 0, 0

        transitions = set()
        for state_symbol in state_symbol_to_states:
            states = state_symbol_to_states[state_symbol]
            q1, a = state_symbol

            if len(states) == 1:
                transitions.add((q1, a, next(iter(states))))
            else:
                count_non_deterministic += 1
                counts = Counter(states)
                first_two_most_common = counts.most_common(2)

                if len(first_two_most_common) > 1:
                    first_most_common, second_most_common = first_two_most_common
                    first_most_common_state, first_most_common_count = first_most_common
                    second_most_common_state, second_most_common_count = second_most_common
                    if first_most_common_count == second_most_common_count:
                        # TODO should we add all transitions with top count and determinize the automaton?
                        count_arbitrarily += 1
                    transitions.add((q1, a, first_most_common_state))
                else:
                    first_most_common = first_two_most_common[0]
                    first_most_common_state, first_most_common_count = first_most_common
                    transitions.add((q1, a, first_most_common_state))

        print("Number of not deterministic transitions", count_non_deterministic)
        print("Number of arbitrarily selected transitions", count_arbitrarily)

        start_state = torch.zeros((1, hidden_size), device=next(model.parameters()).device)
        is_final = model.linear(torch.cat((start_state, model.rnn.states))).argmax(dim=1).cpu().tolist()

        automaton = Automaton(initial_state, transitions, is_final)
        return automaton
