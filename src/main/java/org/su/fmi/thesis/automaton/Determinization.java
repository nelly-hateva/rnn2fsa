package org.su.fmi.thesis.automaton;

import org.su.fmi.thesis.automaton.utils.Enumerable;
import org.su.fmi.thesis.automaton.utils.Enumerator;
import org.su.fmi.thesis.automaton.utils.FNV1A;
import org.su.fmi.thesis.automaton.utils.IntSequence;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.su.fmi.thesis.automaton.Automaton.NO;

public class Determinization implements Enumerable<int[]> {
    private IntSequence setsStates = new IntSequence();
    private IntSequence states = new IntSequence();
    private Enumerator<int[]> num;
    private Automaton a;
    private int firstSet = 0;
    private Set<Integer> set = new HashSet<>();
    private IntSequence heap = new IntSequence();

    public Determinization(Automaton a) {
        this.num = new Enumerator<>(this);
        this.a = a;
    }

    public void reset() {
        setsStates.length = 0;
        states.length = 0;
        num.reset();
        firstSet = 0;
        set.clear();
        heap.length = 0;
    }

    public void determinize(Automaton result) {
        for (int i = 0; i < a.initialStates.length; i++) {
            this.set.add(a.initialStates.seq[i]);
        }
        result.initialStates.add(push());
        while (!queueIsEmpty()) {
            int set = pop();
            result.addState(set);
            if (containsFinalState(set)) {
                result.setStateFinality(set, 1);
            } else {
                result.setStateFinality(set, 0);
            }
            addTransitions(set);
            int tr, letter = NO;
            while ((tr = getNextTransition()) != NO) {
                if (letter == NO) {
                    letter = a.transitionsLabel.seq[tr];
                    this.set.clear();
                } else if (letter != a.transitionsLabel.seq[tr]) {
                    result.addTransition(set, letter, push());
                    letter = a.transitionsLabel.seq[tr];
                    this.set.clear();
                }
                this.set.add(a.transitionsTo.seq[tr]);
            }
            if (letter != NO) {
                result.addTransition(set, letter, push());
            }
        }
        result.initStatesTransitions();
    }

    Automaton determinize() {
        Automaton result = new Automaton();
        determinize(result);
        return result;
    }

    private int push() {
        int[] seq = new int[set.size()];
        int i = 0;
        for (Integer state : set) {
            seq[i] = state;
            i++;
        }
        Arrays.sort(seq);
        return num.add(seq);
    }

    private boolean queueIsEmpty() {
        return firstSet == getNumberOfSets();
    }

    private int pop() {
        int set = firstSet;
        firstSet++;
        return set;
    }

    private void addTransitions(int set) {
        int numberOfStates = getNumberOfStates(set);
        for (int i = 0; i < numberOfStates; i++) {
            int state = states.seq[setsStates.seq[set] + i];
            if (a.getStateNumberOfTransitions(state) != 0) {
                heapPush(a.statesTransitions.seq[state]);
            }
        }
    }

    private int getNextTransition() {
        if (heap.length == 0) {
            return NO;
        }
        int tr = heap.seq[0];
        int stateFrom = a.transitionsFrom.seq[tr];
        if (tr + 1 < a.statesTransitions.seq[stateFrom] + a.getStateNumberOfTransitions(stateFrom)) {
            heap.seq[0]++;
            heapSink();
            return tr;
        }
        heap.length--;
        if (heap.length == 0) {
            return tr;
        }
        heap.seq[0] = heap.seq[heap.length];
        heapSink();
        return tr;
    }

    private int getNumberOfSets() {
        return setsStates.length;
    }

    private int getNumberOfStates(int set) {
        return (set + 1 < getNumberOfSets() ? setsStates.seq[set + 1] : states.length) - setsStates.seq[set];
    }

    private boolean containsFinalState(int set) {
        int numberOfStates = getNumberOfStates(set);
        for (int i = 0; i < numberOfStates; i++) {
            if (a.getStateFinality(states.seq[setsStates.seq[set] + i]) != 0) {
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean equal(int[] seq, int set) {
        if (seq.length != getNumberOfStates(set)) {
            return false;
        }
        for (int i = 0; i < seq.length; i++) {
            if (seq[i] != states.seq[setsStates.seq[set] + i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public long code(int[] seq) {
        int code = FNV1A.OFFSET_BASIS;
        for (int n : seq) {
            code = FNV1A.codeAddInt(code, n);
        }
        return FNV1A.finalizeCode(code);
    }

    @Override
    public long codeByIndex(int set) {
        int code = FNV1A.OFFSET_BASIS;
        int numberOfStates = getNumberOfStates(set);
        for (int i = 0; i < numberOfStates; i++) {
            code = FNV1A.codeAddInt(code, states.seq[setsStates.seq[set] + i]);
        }
        return FNV1A.finalizeCode(code);
    }

    @Override
    public int newObject(int[] seq) {
        int set = getNumberOfSets();
        setsStates.add(states.length);
        states.append(seq);
        return set;
    }

    private int cmpTransitions(int t1, int t2) {
        return Integer.compare(a.transitionsLabel.seq[t1], a.transitionsLabel.seq[t2]);
    }

    private void heapPush(int tr) {
        int p, c = heap.length;
        heap.add(tr);
        while (c > 0) {
            p = (c - 1) / 2;
            if (cmpTransitions(heap.seq[c], heap.seq[p]) < 0) {
                int tmp = heap.seq[c];
                heap.seq[c] = heap.seq[p];
                heap.seq[p] = tmp;
                c = p;
            } else {
                break;
            }
        }
    }

    private void heapSink() {
        int c, l, r;

        c = 0;
        while (true) {
            l = 2 * c + 1;
            if (l < heap.length) {
                r = l + 1;
                if (r < heap.length) {
                    if (cmpTransitions(heap.seq[l], heap.seq[r]) < 0) {
                        if (cmpTransitions(heap.seq[l], heap.seq[c]) < 0) {
                            int tmp = heap.seq[l];
                            heap.seq[l] = heap.seq[c];
                            heap.seq[c] = tmp;
                            c = l;
                        } else {
                            break;
                        }
                    } else if (cmpTransitions(heap.seq[c], heap.seq[r]) > 0) {
                        int tmp = heap.seq[r];
                        heap.seq[r] = heap.seq[c];
                        heap.seq[c] = tmp;
                        c = r;
                    } else {
                        break;
                    }
                } else if (cmpTransitions(heap.seq[l], heap.seq[c]) < 0) {
                    int tmp = heap.seq[l];
                    heap.seq[l] = heap.seq[c];
                    heap.seq[c] = tmp;
                    c = l;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
}
