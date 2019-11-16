package org.su.fmi.thesis.automaton;

import org.su.fmi.thesis.automaton.utils.IntSequence;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;

public class Automaton {

    public static final int NO = Integer.MIN_VALUE;
    protected IntSequence initialStates;
    protected IntSequence transitionsFrom;
    protected IntSequence transitionsLabel;
    protected IntSequence transitionsTo;
    protected IntSequence statesTransitions;
    protected IntSequence statesFinality;

    public Automaton() {
        initialStates = new IntSequence();
        transitionsFrom = new IntSequence();
        transitionsLabel = new IntSequence();
        transitionsTo = new IntSequence();
        statesTransitions = new IntSequence();
        statesFinality = new IntSequence();
    }

    public Automaton(DataInputStream inputStream) throws IOException {
        initialStates = new IntSequence(inputStream);
        transitionsFrom = new IntSequence(inputStream);
        transitionsLabel = new IntSequence(inputStream);
        transitionsTo = new IntSequence(inputStream);
        statesTransitions = new IntSequence(inputStream);
        statesFinality = new IntSequence(inputStream);
    }

    public void write(DataOutputStream outputStream) throws IOException {
        initialStates.write(outputStream);
        transitionsFrom.write(outputStream);
        transitionsLabel.write(outputStream);
        transitionsTo.write(outputStream);
        statesTransitions.write(outputStream);
        statesFinality.write(outputStream);
    }

    public IntSequence getTransitionsFrom() {
        return transitionsFrom;
    }

    public IntSequence getTransitionsLabel() {
        return transitionsLabel;
    }

    public IntSequence getTransitionsTo() {
        return transitionsTo;
    }

    public IntSequence getStatesTransitions() {
        return statesTransitions;
    }

    public IntSequence getStatesFinality() {
        return statesFinality;
    }

    public int getNumberOfStates() {
        return statesTransitions.length;
    }

    public int getNumberOfTransitions() {
        return transitionsTo.length;
    }

    public int getStateFinality(int state) {
        return statesFinality.seq[state];
    }

    public void setStateFinality(int state, int finality) {
        statesFinality.seq[state] = finality;
    }

    /***
     * Expects sorted automaton.
     */
    public int getStateNumberOfTransitions(int state) {
        return ((state + 1 < getNumberOfStates()) ? statesTransitions.seq[state + 1] : getNumberOfTransitions()) - statesTransitions.seq[state];
    }

    public int getNumberOfFinalStates() {
        int c = 0;
        for (int i = 0; i < statesFinality.seq.length; ++i) {
            if (statesFinality.seq[i] == 1) {
                ++c;
            }
        }
        return c;
    }

    public IntSequence getInitialStates() {
        return initialStates;
    }

    public void addState(int state) {
        int numberOfStates = getNumberOfStates();
        if (numberOfStates <= state) {
            for (int i = numberOfStates; i <= state; i++) {
                statesTransitions.add(NO);
                statesFinality.add(0);
            }
        }
    }

    public void addInitialState(int state) {
        initialStates.add(state);
    }

    public void addTransition(int stateFrom, int label, int stateTo) {
        int numberOfTransitions = getNumberOfTransitions();
        addState(Math.max(stateFrom, stateTo));
        transitionsFrom.add(stateFrom);
        transitionsLabel.add(label);
        transitionsTo.add(stateTo);
        if (statesTransitions.seq[stateFrom] == NO) {
            statesTransitions.seq[stateFrom] = numberOfTransitions;
        }
    }

    /***
     * Expects sorted deterministic automaton.
     */
    public int delta(int state, int label) {
        int tr = getTransition(state, label);
        return (tr != NO) ? transitionsTo.seq[tr] : NO;
    }

    /***
     * Expects sorted deterministic automaton.
     */
    public int getTransition(int state, int label) {
        int numberOfTransitions = getStateNumberOfTransitions(state);
        if (numberOfTransitions == 0) {
            return NO;
        }
        int tr = Arrays.binarySearch(transitionsLabel.seq, statesTransitions.seq[state], statesTransitions.seq[state] + numberOfTransitions, label);
        return (tr < 0) ? NO : tr;
    }

    public void sort() {
        Integer[] transitions = new Integer[getNumberOfTransitions()];
        for (int i = 0; i < transitions.length; i++) {
            transitions[i] = i;
        }
        Arrays.sort(transitions, (t1, t2) -> {
            if (transitionsFrom.seq[t1] < transitionsFrom.seq[t2]) {
                return -1;
            }
            if (transitionsFrom.seq[t1] > transitionsFrom.seq[t2]) {
                return 1;
            }
            if (transitionsLabel.seq[t1] < transitionsLabel.seq[t2]) {
                return -1;
            }
            if (transitionsLabel.seq[t1] > transitionsLabel.seq[t2]) {
                return 1;
            }
            if (transitionsTo.seq[t1] < transitionsTo.seq[t2]) {
                return -1;
            }
            if (transitionsTo.seq[t1] > transitionsTo.seq[t2]) {
                return 1;
            }
            return 0;
        });
        for (int i = 0; i < transitions.length; i++) {
            if (transitions[i] != NO) {
                int from = transitionsFrom.seq[i];
                int label = transitionsLabel.seq[i];
                int to = transitionsTo.seq[i];
                int j = i;
                while (transitions[j] != i) {
                    transitionsFrom.seq[j] = transitionsFrom.seq[transitions[j]];
                    transitionsLabel.seq[j] = transitionsLabel.seq[transitions[j]];
                    transitionsTo.seq[j] = transitionsTo.seq[transitions[j]];
                    int next = transitions[j];
                    transitions[j] = NO;
                    j = next;
                }
                transitionsFrom.seq[j] = from;
                transitionsLabel.seq[j] = label;
                transitionsTo.seq[j] = to;
                transitions[j] = NO;
            }
        }
        initStatesTransitions();
    }

    public void initStatesTransitions() {
        int numberOfStates = getNumberOfStates();
        int numberOfTransitions = getNumberOfTransitions();
        if (numberOfTransitions == 0) {
            for (int s = 0; s < numberOfStates; s++) {
                statesTransitions.seq[s] = 0;
            }
            return;
        }
        int state = transitionsFrom.seq[0];
        for (int s = 0; s <= state; s++) {
            statesTransitions.seq[s] = 0;
        }
        for (int i = 1; i < numberOfTransitions; i++) {
            int prevState = transitionsFrom.seq[i - 1];
            state = transitionsFrom.seq[i];
            if (prevState != state) {
                for (int s = prevState + 1; s <= state; s++) {
                    statesTransitions.seq[s] = i;
                }
            }
        }
        for (int s = state + 1; s < numberOfStates; s++) {
            statesTransitions.seq[s] = numberOfTransitions;
        }
    }

    /***
     * Expects sorted automaton.
     */
    public Automaton determinize() {
        return new Determinization(this).determinize();
    }

    /***
     * Expects sorted trimmed automaton.
     */
    public Automaton minimize() {
        return new Minimization(this).minimize();
    }

    public Automaton reverse() {
        Automaton result = new Automaton();
        reverse(result);
        return result;
    }

    public void reverse(Automaton result) {
        int numberOfStates = getNumberOfStates();
        for (int q = 0; q < numberOfStates; q++) {
            result.addState(q);
            if (statesFinality.seq[q] != 0) {
                result.initialStates.add(q);
            }
        }
        for (int i = 0; i < initialStates.length; i++) {
            result.setStateFinality(initialStates.seq[i], 1);
        }
        int numberOfTransitions = getNumberOfTransitions();
        for (int i = 0; i < numberOfTransitions; i++) {
            result.addTransition(transitionsTo.seq[i], transitionsLabel.seq[i], transitionsFrom.seq[i]);
        }
        result.sort();
    }

    /***
     * Expects sorted deterministic automata.
     */
    public static boolean isomorphic(Automaton a1, Automaton a2) {
        int numberOfStates = a1.getNumberOfStates();
        if (numberOfStates != a2.getNumberOfStates() ||
                a1.getNumberOfTransitions() != a2.getNumberOfTransitions() ||
                a1.initialStates.length != a2.initialStates.length) {
            return false;
        }
        if (a1.initialStates.length == 0) {
            return true;
        }
        int[] a1Toa2 = new int[numberOfStates];
        int[] a2Toa1 = new int[numberOfStates];
        for (int i = 0; i < numberOfStates; i++) {
            a1Toa2[i] = a2Toa1[i] = NO;
        }
        Set<Integer> visited = new HashSet<>(numberOfStates);
        int q1 = a1.initialStates.seq[0];
        int q2 = a2.initialStates.seq[0];
        a1Toa2[q1] = q2;
        a2Toa1[q2] = q1;
        IntSequence queue = new IntSequence(numberOfStates, -1);
        queue.add(q1);
        visited.add(q1);
        int head = 0;
        while (head != queue.length) {
            q1 = queue.seq[head];
            head++;
            q2 = a1Toa2[q1];
            if (q2 == NO || a1.getStateFinality(q1) != a2.getStateFinality(q2)) {
                return false;
            }
            int numberOfTransitions = a1.getStateNumberOfTransitions(q1);
            if (a2.getStateNumberOfTransitions(q2) != numberOfTransitions) {
                return false;
            }
            for (int i = 0; i < numberOfTransitions; i++) {
                int tr1 = a1.statesTransitions.seq[q1] + i;
                int tr2 = a2.statesTransitions.seq[q2] + i;
                if (a1.transitionsLabel.seq[tr1] != a2.transitionsLabel.seq[tr2]) {
                    return false;
                }
                int to1 = a1.transitionsTo.seq[tr1];
                int to2 = a2.transitionsTo.seq[tr2];
                if ((a1Toa2[to1] == NO && a2Toa1[to2] != NO) || (a1Toa2[to1] != NO && a2Toa1[to2] == NO)) {
                    return false;
                }
                if (a1Toa2[to1] == NO) {
                    a1Toa2[to1] = to2;
                    a2Toa1[to2] = to1;
                } else if (a1Toa2[to1] != to2 || a2Toa1[to2] != to1) {
                    return false;
                }
                if (!visited.contains(to1)) {
                    queue.add(to1);
                    visited.add(to1);
                }
            }
        }
        return true;
    }

    public Automaton plus() {
        Automaton result = unsortedPlus();
        result.sort();
        return result;
    }

    public Automaton star() {
        Automaton result = unsortedPlus();
        int q = result.getNumberOfStates();
        result.addState(q);
        result.addInitialState(q);
        result.setStateFinality(q, 1);
        result.sort();
        return result;
    }

    public static Automaton concat(Automaton a1, Automaton a2) {
        boolean a2AcceptsEpsilon = acceptsEpsilon(a2);

        Automaton result = new Automaton();
        int a1NumberOfStates = a1.getNumberOfStates();
        for (int q1 = 0; q1 < a1NumberOfStates; q1++) {
            result.addState(q1);
            if (a2AcceptsEpsilon && a1.getStateFinality(q1) != 0) {
                result.setStateFinality(q1, a1.getStateFinality(q1));
            }
        }
        result.initialStates.cpy(a1.initialStates);

        int a1NumberOfTransitions = a1.getNumberOfTransitions();
        for (int t1 = 0; t1 < a1NumberOfTransitions; t1++) {
            result.addTransition(a1.transitionsFrom.seq[t1], a1.transitionsLabel.seq[t1], a1.transitionsTo.seq[t1]);
        }

        int a2NumberOfStates = a2.getNumberOfStates();
        for (int q2 = 0; q2 < a2NumberOfStates; q2++) {
            result.addState(a1NumberOfStates + q2);
            result.setStateFinality(a1NumberOfStates + q2, a2.getStateFinality(q2));
        }

        int a2NumberOfTransitions = a2.getNumberOfTransitions();
        for (int t2 = 0; t2 < a2NumberOfTransitions; t2++) {
            result.addTransition(a1NumberOfStates + a2.transitionsFrom.seq[t2], a2.transitionsLabel.seq[t2], a1NumberOfStates + a2.transitionsTo.seq[t2]);
        }

        for (int q1 = 0; q1 < a1NumberOfStates; q1++) {
            if (a1.getStateFinality(q1) != 0) {
                for (int i = 0; i < a2.initialStates.length; i++) {
                    int i2 = a2.initialStates.seq[i];
                    int i2NumberOfTransitions = a2.getStateNumberOfTransitions(i2);
                    for (int j = 0; j < i2NumberOfTransitions; j++) {
                        int tr = a2.statesTransitions.seq[i2] + j;
                        result.addTransition(q1, a2.transitionsLabel.seq[tr], a1NumberOfStates + a2.transitionsTo.seq[tr]);
                    }
                }
            }
        }

        result.sort();
        return result;
    }

    public static Automaton union(Automaton a1, Automaton a2) {
        Automaton result = new Automaton();
        int a1NumberOfStates = a1.getNumberOfStates();
        for (int q1 = 0; q1 < a1NumberOfStates; q1++) {
            result.addState(q1);
            result.setStateFinality(q1, a1.getStateFinality(q1));
        }
        result.initialStates.cpy(a1.initialStates);

        int a1NumberOfTransitions = a1.getNumberOfTransitions();
        for (int t1 = 0; t1 < a1NumberOfTransitions; t1++) {
            result.addTransition(a1.transitionsFrom.seq[t1], a1.transitionsLabel.seq[t1], a1.transitionsTo.seq[t1]);
        }

        int a2NumberOfStates = a2.getNumberOfStates();
        for (int q2 = 0; q2 < a2NumberOfStates; q2++) {
            result.addState(a1NumberOfStates + q2);
            result.setStateFinality(a1NumberOfStates + q2, a2.getStateFinality(q2));
        }

        int a2NumberOfTransitions = a2.getNumberOfTransitions();
        for (int t2 = 0; t2 < a2NumberOfTransitions; t2++) {
            result.addTransition(a1NumberOfStates + a2.transitionsFrom.seq[t2], a2.transitionsLabel.seq[t2], a1NumberOfStates + a2.transitionsTo.seq[t2]);
        }

        for (int i = 0; i < a2.initialStates.length; i++) {
            result.addInitialState(a1NumberOfStates + a2.initialStates.seq[i]);
        }

        result.sort();
        return result;
    }

    private static boolean acceptsEpsilon(Automaton a) {
        IntSequence initialStates = a.getInitialStates();
        for (int i = 0; i < initialStates.length; i++) {
            if (a.getStateFinality(initialStates.seq[i]) != 0) {
                return true;
            }
        }
        return false;
    }

    private Automaton unsortedPlus() {
        Automaton result = new Automaton();
        int numberOfStates = getNumberOfStates();
        for (int q = 0; q < numberOfStates; q++) {
            result.addState(q);
            result.setStateFinality(q, getStateFinality(q));
        }
        result.initialStates.append(initialStates);

        int numberOfTransitions = getNumberOfTransitions();
        for (int t = 0; t < numberOfTransitions; t++) {
            int from = transitionsFrom.seq[t];
            int label = transitionsLabel.seq[t];
            int to = transitionsTo.seq[t];
            result.addTransition(from, label, to);
            if (getStateFinality(to) != 0) {
                for (int j = 0; j < initialStates.length; j++) {
                    result.addTransition(from, label, initialStates.seq[j]);
                }
            }
        }
        return result;
    }

    public void clear() {
        initialStates.length = 0;
        transitionsFrom.length = 0;
        transitionsLabel.length = 0;
        transitionsTo.length = 0;
        statesTransitions.length = 0;
        statesFinality.length = 0;
    }

    public int accepts(IntSequence seq) {
        int state = initialStates.seq[0];
        for (int i = 0; i < seq.length; ++i) {
            int next_state = delta(state, seq.seq[i]);
            if (next_state != NO) {
                state = next_state;
            } else {
                return 0;
            }
        }
        return statesFinality.seq[state];
    }

    public int numberOfReachableStates() {
        boolean[] visited = new boolean[getNumberOfStates()];
        Deque<Integer> stack = new ArrayDeque<>(getNumberOfStates());

        for (int i = 0; i < initialStates.length; ++i) {
            stack.push(initialStates.seq[i]);
        }

        dfs(stack, visited);

        int count = 0;
        for (int i = 0; i < getNumberOfStates(); ++i) {
            if (visited[i]) {
                ++count;
            }
        }
        return count;
    }

    private void dfs(Deque<Integer> stack, boolean[] visited) {
        while (!stack.isEmpty()) {
            int top = stack.pop();
            visited[top] = true;

            for (int i = 0; i < getStateNumberOfTransitions(top); ++i) {
                int adjacent = transitionsTo.seq[statesTransitions.seq[top] + i];
                if (!visited[adjacent]) {
                    stack.push(adjacent);
                }
            }
        }
    }

    public int numberOfCoReachableStates() {
        int count = 0;
        for (int state = 0; state < statesFinality.length; ++state) {
            if (coReachable(state)) {
                ++count;
            }
        }
        return count;
    }

    private boolean coReachable(int state) {
        boolean[] visited = new boolean[getNumberOfStates()];
        Deque<Integer> stack = new ArrayDeque<>(getNumberOfStates());

        stack.push(state);

        while (!stack.isEmpty()) {
            int top = stack.pop();
            visited[top] = true;

            if (statesFinality.seq[top] == 1) {
                return true;
            }

            for (int i = 0; i < getStateNumberOfTransitions(top); ++i) {
                int adjacent = transitionsTo.seq[statesTransitions.seq[top] + i];
                if (!visited[adjacent]) {
                    stack.push(adjacent);
                }
            }
        }
        return false;
    }

    public boolean containsCycle() {
        boolean[] visited = new boolean[getNumberOfStates()];
        Deque<Integer> stack = new ArrayDeque<>(getNumberOfStates());

        for (int i = 0; i < initialStates.length; ++i) {
            stack.push(initialStates.seq[i]);
        }

        return !containsCycle(stack, visited);
    }

    private boolean containsCycle(Deque<Integer> stack, boolean[] visited) {
        while (!stack.isEmpty()) {
            int top = stack.pop();
            visited[top] = true;

            for (int i = 0; i < getStateNumberOfTransitions(top); ++i) {
                int adjacent = transitionsTo.seq[statesTransitions.seq[top] + i];
                if (!visited[adjacent]) {
                    stack.push(adjacent);
                } else {
                    return true;
                }
            }
        }
        return false;
    }
}
