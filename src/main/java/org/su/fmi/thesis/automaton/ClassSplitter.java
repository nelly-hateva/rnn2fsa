package org.su.fmi.thesis.automaton;

import org.su.fmi.thesis.automaton.utils.Enumerable;
import org.su.fmi.thesis.automaton.utils.Enumerator;
import org.su.fmi.thesis.automaton.utils.FNV1A;
import org.su.fmi.thesis.automaton.utils.IntSequence;

class ClassSplitter implements Enumerable<Integer> {
    private Minimization minimization;
    private IntSequence states = new IntSequence();
    private int transition;
    private int[] statesClassNumber;
    private Automaton a;
    private Enumerator<Integer> num;

    ClassSplitter(Minimization minimization) {
        this.minimization = minimization;
        this.statesClassNumber = minimization.statesClassNumber;
        this.a = minimization.a;
        this.num = new Enumerator<>(this, true);
    }

    void reset(int transition) {
        this.transition = transition;
        num.reset();
        states.length = 0;
    }

    int add(int state) {
        return num.add(state);
    }

    int getNumberOfClasses() {
        return states.length;
    }

    private int delta(int state) {
        return a.transitionsTo.seq[a.statesTransitions.seq[state] + transition];
    }

    @Override
    public boolean equal(Integer state, int index) {
        return statesClassNumber[delta(states.seq[index])] == statesClassNumber[delta(state)];
    }

    @Override
    public long code(Integer state) {
        return FNV1A.finalizeCode(FNV1A.codeAddInt(FNV1A.OFFSET_BASIS, statesClassNumber[delta(state)]));
    }

    @Override
    public long codeByIndex(int index) {
        return code(states.seq[index]);
    }

    @Override
    public int newObject(Integer state) {
        int index = states.length;
        states.add(state);
        return index;
    }
}
