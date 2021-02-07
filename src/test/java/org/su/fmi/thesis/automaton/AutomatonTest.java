package org.su.fmi.thesis.automaton;

import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AutomatonTest {

    @Test
    void test() throws IOException {
        Automaton a = new Automaton();

        int initial_state = 0;
        a.addInitialState(initial_state);
        int states_count = 1;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                Objects.requireNonNull(this.getClass().getClassLoader().getResourceAsStream("numeral8.dat"))
        ))) {
            String line;

            while ((line = br.readLine()) != null) {
                line = line.toLowerCase();
                int prev_state = initial_state;
                for (int i = 0; i < line.length(); i++) {
                    a.addTransition(prev_state, line.charAt(i), states_count);
                    if (i == line.length() - 1) {
                        a.setStateFinality(states_count, 1);
                    }
                    prev_state = states_count;
                    states_count++;
                }
            }
        }

        a.sort();
        a = a.determinize();
        a = a.minimize();

        assertEquals(13, a.getNumberOfStates());
        assertEquals(15, a.getNumberOfTransitions());
        assertEquals(5, a.getNumberOfFinalStates());
    }

    @Test
    void test0() throws IOException {
        Automaton a = new Automaton();

        int initial_state = 0;
        a.addInitialState(initial_state);
        int states_count = 1;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                Objects.requireNonNull(this.getClass().getClassLoader().getResourceAsStream("numeral.dat"))
        ))) {
            String line;

            while ((line = br.readLine()) != null) {
                line = line.toLowerCase();
                int prev_state = initial_state;
                for (int i = 0; i < line.length(); i++) {
                    a.addTransition(prev_state, line.charAt(i), states_count);
                    if (i == line.length() - 1) {
                        a.setStateFinality(states_count, 1);
                    }
                    prev_state = states_count;
                    states_count++;
                }
            }
        }

        a.sort();
        a = a.determinize();
        a = a.minimize();

        assertEquals(133, a.getNumberOfStates());
        assertEquals(221, a.getNumberOfTransitions());
        assertEquals(21, a.getNumberOfFinalStates());
    }

    @Test
    void test1() throws IOException {
        Automaton a = new Automaton();

        int initial_state = 0;
        a.addInitialState(initial_state);
        int states_count = 1;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                Objects.requireNonNull(this.getClass().getClassLoader().getResourceAsStream("words.dat"))
        ))) {
            String line;

            while ((line = br.readLine()) != null) {
                line = line.toLowerCase();
                int prev_state = initial_state;
                for (int i = 0; i < line.length(); i++) {
                    a.addTransition(prev_state, line.charAt(i), states_count);
                    if (i == line.length() - 1) {
                        a.setStateFinality(states_count, 1);
                    }
                    prev_state = states_count;
                    states_count++;
                }
            }
        }

        a.sort();
        a = a.determinize();
        a = a.minimize();

        assertEquals(30568, a.getNumberOfStates());
        assertEquals(77838, a.getNumberOfTransitions());
        assertEquals(4826, a.getNumberOfFinalStates());
    }

    @Test
    void test2() throws IOException {
        Automaton a = new Automaton();

        int initial_state = 0;
        a.addInitialState(initial_state);
        int states_count = 1;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                Objects.requireNonNull(this.getClass().getClassLoader().getResourceAsStream("train.data"))
        ))) {
            String line;

            while ((line = br.readLine()) != null) {
                line = line.toLowerCase();
                int prev_state = initial_state;
                for (int i = 0; i < line.length(); i++) {
                    a.addTransition(prev_state, line.charAt(i), states_count);
                    if (i == line.length() - 1) {
                        a.setStateFinality(states_count, 1);
                    }
                    prev_state = states_count;
                    states_count++;
                }
            }
        }

        a.sort();
        a = a.determinize();
        a = a.minimize();

        assertEquals(75477, a.getNumberOfStates());
        assertEquals(232603, a.getNumberOfTransitions());
        assertEquals(17643, a.getNumberOfFinalStates());
    }
}
