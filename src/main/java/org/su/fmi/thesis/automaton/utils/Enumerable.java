package org.su.fmi.thesis.automaton.utils;

public interface Enumerable<T> {
    boolean equal(T obj, int index);

    long code(T obj);

    long codeByIndex(int index);

    int newObject(T obj);
}
