package org.su.fmi.thesis.automaton;

import org.su.fmi.thesis.automaton.utils.Enumerable;
import org.su.fmi.thesis.automaton.utils.Enumerator;
import org.su.fmi.thesis.automaton.utils.FNV1A;
import org.su.fmi.thesis.automaton.utils.IntSequence;

import static org.su.fmi.thesis.automaton.Automaton.NO;

class Minimization implements Enumerable<Integer> {
    int[] statesClassNumber;
    private int[] statesNewClassNumber;
    private int[] statesNext;
    private IntSequence classesFirstState = new IntSequence();
    private Enumerator<Integer> num;
    Automaton a;
    private ClassSplitter classSplitter;

    Minimization(Automaton a) {
        this.a = a;
        this.num = new Enumerator<>(this);
        this.statesClassNumber = new int[a.getNumberOfStates()];
        this.statesNext = new int[a.getNumberOfStates()];
        this.statesNewClassNumber = new int[a.getNumberOfStates()];
        this.classSplitter = new ClassSplitter(this);
    }

    @Override
    public boolean equal(Integer state, int cl) {
        int q = classesFirstState.seq[cl];
        int numberOfTransitions = a.getStateNumberOfTransitions(q);
        if (numberOfTransitions != a.getStateNumberOfTransitions(state) || a.getStateFinality(q) != a.getStateFinality(state)) {
            return false;
        }
        for (int i = 0; i < numberOfTransitions; i++) {
            if (a.transitionsLabel.seq[a.statesTransitions.seq[q] + i] != a.transitionsLabel.seq[a.statesTransitions.seq[state] + i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public long code(Integer state) {
        int code = FNV1A.codeAddInt(FNV1A.OFFSET_BASIS, a.getStateFinality(state));
        int numberOfTransitions = a.getStateNumberOfTransitions(state);
        for (int i = 0; i < numberOfTransitions; i++) {
            code = FNV1A.codeAddInt(code, a.transitionsLabel.seq[a.statesTransitions.seq[state] + i]);
        }
        return FNV1A.finalizeCode(code);
    }

    @Override
    public long codeByIndex(int cl) {
        return code(classesFirstState.seq[cl]);
    }

    @Override
    public int newObject(Integer state) {
        int cl = addClass();
        addStateToClass(cl, state);
        return cl;
    }

    private int getNumberOfClasses() {
        return classesFirstState.length;
    }

    private int addClass() {
        int n = getNumberOfClasses();
        classesFirstState.add(NO);
        return n;
    }

    private void addStateToClass(int cl, int state) {
        statesNext[state] = classesFirstState.seq[cl];
        classesFirstState.seq[cl] = state;
        statesClassNumber[state] = cl;
    }

    Automaton minimize() {
        if (a.initialStates.length == 0) {
            return new Automaton();
        }
        int numberOfStates = a.getNumberOfStates();
        for (int state = 0; state < numberOfStates; state++) {
            int numberOfClasses = getNumberOfClasses();
            int cl = num.add(state);
            if (cl < numberOfClasses) {
                addStateToClass(cl, state);
            }
        }
        int cl = 0;
        int checked = 0;
        while (checked < getNumberOfClasses()) {
            if (splitClass(cl)) {
                checked = 0;
            } else {
                checked++;
            }
            cl = (cl + 1) % getNumberOfClasses();
        }
        Automaton result = new Automaton();
        result.initialStates.add(statesClassNumber[a.initialStates.seq[0]]);
        int numberOfClasses = getNumberOfClasses();
        for (cl = 0; cl < numberOfClasses; cl++) {
            result.addState(cl);
            int q = classesFirstState.seq[cl];
            result.setStateFinality(cl, a.getStateFinality(q));
            int numberOfTransitions = a.getStateNumberOfTransitions(q);
            for (int i = 0; i < numberOfTransitions; i++) {
                int tr = a.statesTransitions.seq[q] + i;
                result.addTransition(cl, a.transitionsLabel.seq[tr], statesClassNumber[a.transitionsTo.seq[tr]]);
            }
        }
        result.initStatesTransitions();
        return result;
    }

    private boolean splitClass(int cl) {
        if (statesNext[classesFirstState.seq[cl]] == NO) {
            return false;
        }
        int numberOfTransitions = a.getStateNumberOfTransitions(classesFirstState.seq[cl]);
        int j;
        for (j = 0; j < numberOfTransitions; j++) {
            classSplitter.reset(j);
            for (int s = classesFirstState.seq[cl]; s != NO; s = statesNext[s]) {
                int c = classSplitter.add(s);
                if (c == 0) {
                    statesNewClassNumber[s] = cl;
                } else {
                    statesNewClassNumber[s] = getNumberOfClasses() + c - 1;
                }
            }
            if (classSplitter.getNumberOfClasses() > 1) {
                int prev = classesFirstState.seq[cl];
                for (int s = statesNext[prev]; s != NO; s = statesNext[prev]) {
                    if (statesNewClassNumber[s] == cl) {
                        prev = s;
                    } else {
                        int newClass = statesNewClassNumber[s];
                        if (getNumberOfClasses() == newClass) {
                            addClass();
                        }
                        statesNext[prev] = statesNext[s];
                        addStateToClass(newClass, s);
                    }
                }
                break;
            }
        }
        return j < numberOfTransitions;
    }
}
