import torch
from torch import nn

from automata import Automaton


class Algorithm2:
    @staticmethod
    def model_to_automaton(model, inv_alphabet, threshold=0):
        hidden_size = model.rnn.rnn_cell.weight_hh.size(0)
        device = next(model.parameters()).device

        initial_state, transitions = 0, set()
        states, queue = set(), [initial_state]

        x = torch.tensor(list(inv_alphabet.keys()), device=device).unsqueeze(1)
        length = torch.ones(len(inv_alphabet), device='cpu')

        while len(queue) != 0:

            current_state = queue.pop(0)
            states.add(current_state)

            if current_state == initial_state:
                h_0 = torch.zeros((len(inv_alphabet), hidden_size), device=device)
            else:
                h_0 = model.rnn.states[current_state - 1].expand(len(inv_alphabet), -1)

            _, transition_probabilities = model(x, length, h_0=h_0, return_probabilities=True)
            transition_probabilities, _ = nn.utils.rnn.pad_packed_sequence(transition_probabilities, batch_first=True)
            probabilities, next_states = torch.max(transition_probabilities[:, 0, ], dim=1)

            next_states += 1

            for i, char in enumerate(inv_alphabet):
                symbol = inv_alphabet[char]
                transitions.add((current_state, symbol, next_states[i].item(), probabilities[i].item()))

            for state in set(next_states.detach().cpu().tolist()):
                if state not in states and state not in queue:
                    queue.append(state)

        mapping = {state: i for i, state in enumerate(states)}
        transitions = {
            (mapping[q1], symbol, mapping[q2])
            for (q1, symbol, q2, probability) in transitions
            if probability >= threshold
        }

        start_state = torch.zeros((1, hidden_size), device=device)
        is_final = model.linear(torch.cat((start_state, model.rnn.states))).argmax(dim=1).cpu().tolist()
        is_final = [is_final[state] for state, _ in mapping.items()]

        automaton = Automaton(initial_state, transitions, is_final)
        return automaton
