package org.su.fmi.thesis.automaton.utils;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

public class IntSequence extends Sequence {
    public int[] seq;

    public IntSequence(int alloced, int growth) {
        super(0, growth);
        seq = new int[alloced];
    }

    public IntSequence() {
        this(63, -1);
    }

    public IntSequence(DataInputStream instr) throws IOException {
        super(instr);
        seq = new int[length];
        for (int i = 0; i < length; i++) {
            seq[i] = instr.readInt();
        }
    }

    public void add(int n) {
        if (length == seq.length) {
            seq = Arrays.copyOf(seq, getNewAlloced());
        }
        seq[length] = n;
        length++;
    }

    @Override
    public void write(DataOutputStream outstr) throws IOException {
        super.write(outstr);
        for (int i = 0; i < length; i++) {
            outstr.writeInt(seq[i]);
        }
        outstr.flush();
    }

    public void append(int[] array) {
        if (array.length > 0) {
            int newLength = length + array.length;
            if (seq.length < newLength) {
                seq = Arrays.copyOf(seq, getNewAlloced(newLength));
            }
            System.arraycopy(array, 0, seq, length, array.length);
            length = newLength;
        }
    }

    public void append(IntSequence sequence) {
        if (sequence.length > 0) {
            int newLength = length + sequence.length;
            if (seq.length < newLength) {
                seq = Arrays.copyOf(seq, getNewAlloced(newLength));
            }
            System.arraycopy(sequence.seq, 0, seq, length, sequence.length);
            length = newLength;
        }
    }

    public boolean contains(int n) {
        for (int i = 0; i < length; i++) {
            if (seq[i] == n) {
                return true;
            }
        }
        return false;
    }

    public void cpy(IntSequence src) {
        if (seq.length < src.length) {
            seq = Arrays.copyOf(seq, getNewAlloced(src.length));
        }
        System.arraycopy(src.seq, 0, seq, 0, src.length);
        length = src.length;
    }
}
