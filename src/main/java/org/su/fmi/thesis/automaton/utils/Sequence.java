package org.su.fmi.thesis.automaton.utils;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class Sequence {
    public int length;
    public int growth;
    /*
     * This class provides dynamic array functionality.
     * The actual array should be defined in a class that extends this one.
     * The member length is the length of the array.
     * The growth member determines how the array is resized.
     * If growth > 0 then the allocated memory is increased by growth positions.
     * If growth < 0 then the allocated memory is increased by -allocated/growth positions.
     * For example if growth equals -1 then the allocated memory is doubled when needed to be resized.
     */

    public Sequence(int length, int growth) {
        this.length = length;
        this.growth = growth;
    }

    public int getNewAlloced() {
        int newAlloced;

        if (growth < 0) {
            newAlloced = length - length / growth;
        } else {
            newAlloced = length + growth;
        }
        if (newAlloced <= length) {
            newAlloced = length + 1;
        }
        return newAlloced;
    }

    public int getNewAlloced(int newLength) {
        int newAlloced;

        if (growth < 0) {
            newAlloced = newLength - newLength / growth;
        } else {
            newAlloced = newLength + growth;
        }
        if (newAlloced < 0) {
            System.out.println(newLength);
        }
        return newAlloced;
    }

    public Sequence(DataInputStream instr) throws IOException {
        length = instr.readInt();
        growth = instr.readInt();
    }

    public void write(DataOutputStream outstr) throws IOException {
        outstr.writeInt(length);
        outstr.writeInt(growth);
        outstr.flush();
    }
}
