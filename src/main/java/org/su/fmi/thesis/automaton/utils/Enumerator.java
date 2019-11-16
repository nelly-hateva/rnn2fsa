package org.su.fmi.thesis.automaton.utils;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class Enumerator<T> {
    public final static int HASH_STEP = 107;
    private int[] hash;
    private Enumerable set;
    private int numberOfObjects;
    private IntSequence nonEmptyPositions;

    /* This class implements an injective mapping from a set of objects to {0,1,2,...,numberOfObjects-1}.
     * The hash array represents an open addressing hash table of the indexes of the strings.
     * The actual storage of the set is provided by Enumerable
     * This class calls
     * 1. equal(T obj, int index) while searching in the hash table
     * 2. code(T obj) while searching in the hash table
     * 3. codeByIndex(int index) while resizing the hash table
     * 4. void newObject(T obj) when a new object is added in the hash table
     */

    /**
     * hashLength must be 2^n-1 for some n > 1,
     * i.e. hashLength must belong to {3,7,15,31,63,127,255,511,1023,2047,...}.
     */
    public Enumerator(Enumerable<T> set, int hashLength, boolean fastReset) {
        this.set = set;
        this.hash = new int[hashLength];
        for (int i = 0; i < hashLength; i++) {
            this.hash[i] = -1;
        }
        this.numberOfObjects = 0;
        if (fastReset) {
            this.nonEmptyPositions = new IntSequence();
        } else {
            this.nonEmptyPositions = null;
        }
    }

    /**
     * hashLength must be 2^n-1 for some n > 1,
     * i.e. hashLength must belong to {3,7,15,31,63,127,255,511,1023,2047,...}.
     */
    public Enumerator(Enumerable<T> set, int hashLength) {
        this(set, hashLength, false);
    }

    public Enumerator(Enumerable<T> set, boolean fastReset) {
        this(set, 63, fastReset);
    }

    public Enumerator(Enumerable<T> set) {
        this(set, 63, false);
    }

    /**
     * Serialization
     */
    public void write(DataOutputStream outstr) throws IOException {
        outstr.writeInt(numberOfObjects);
        outstr.writeInt(hash.length);
        for (int i = 0; i < hash.length; i++) {
            outstr.writeInt(hash[i]);
        }
        outstr.flush();
    }

    /**
     * Deserialization
     */
    public Enumerator(Enumerable<T> set, DataInputStream instr) throws IOException {
        this.set = set;
        numberOfObjects = instr.readInt();
        hash = new int[instr.readInt()];
        for (int i = 0; i < hash.length; i++) {
            hash[i] = instr.readInt();
        }
    }

    /**
     * Adds an object to the set. Returns the number corresponding to the object.
     */
    public int add(T obj) {
        int i, index;
        for (i = (int) (set.code(obj) % ((long) (hash.length))); hash[i] != -1; i = (i + HASH_STEP) % hash.length) {
            if (set.equal(obj, hash[i])) {
                return hash[i];
            }
        }
        hash[i] = index = set.newObject(obj);
        if (nonEmptyPositions != null) {
            nonEmptyPositions.add(i);
        }
        numberOfObjects++;
        if (((long) (9)) * ((long) (hash.length))
                < ((long) (10)) * ((long) (numberOfObjects))) {//if the load factor of the hash table is > 90%
            //resize the hash table
            int[] newHash = new int[2 * hash.length + 1];
            for (i = 0; i < newHash.length; i++) {
                newHash[i] = -1;
            }
            if (nonEmptyPositions != null) {
                nonEmptyPositions.length = 0;
            }
            for (int s = 0; s < hash.length; s++) {
                if (hash[s] != -1) {
                    for (
                            i = (int) (set.codeByIndex(hash[s]) % ((long) (newHash.length)));
                            newHash[i] != -1; i = (i + HASH_STEP) % newHash.length)
                        ;
                    newHash[i] = hash[s];
                    if (nonEmptyPositions != null) {
                        nonEmptyPositions.add(i);
                    }
                }
            }
            hash = newHash;
        }
        return index;
    }

    /**
     * Returns the number corresponding to the given object if the given object belongs to the set.
     * Returns -1 if the given object does not belong to the set.
     */
    public int get(T obj) {
        for (int i = (int) (set.code(obj) % ((long) (hash.length))); hash[i] != -1; i = (i + HASH_STEP) % hash.length) {
            if (set.equal(obj, hash[i])) {
                return hash[i];
            }
        }
        return -1;
    }

    public void reset() {
        if (nonEmptyPositions != null) {
            for (int i = 0; i < nonEmptyPositions.length; i++) {
                hash[nonEmptyPositions.seq[i]] = -1;
            }
            nonEmptyPositions.length = 0;
        } else {
            for (int i = 0; i < hash.length; i++) {
                hash[i] = -1;
            }
        }
        numberOfObjects = 0;
    }

    public void delete(T obj) {
        int i;
        for (i = (int) (set.code(obj) % ((long) (hash.length))); hash[i] != -1; i = (i + HASH_STEP) % hash.length) {
            if (set.equal(obj, hash[i])) {
                break;
            }
        }
        if (hash[i] == -1) {
            return;
        }
        hash[i] = -1;
        int index, j;
        for (i = (i + HASH_STEP) % hash.length; hash[i] != -1; i = (i + HASH_STEP) % hash.length) {
            index = hash[i];
            hash[i] = -1;
            for (j = (int) (set.codeByIndex(index) % ((long) (hash.length))); hash[j] != -1; j = (j + HASH_STEP) % hash.length)
                ;
            hash[j] = index;
        }
        numberOfObjects--;
    }

    public int getHashLength() {
        return hash.length;
    }
}
