package org.su.fmi.thesis.automaton.utils;

/**
 * This class provides functionality for computing fnv1a hash codes.
 */
public class FNV1A {
    public static final int OFFSET_BASIS = 0x811C9DC5;
    public static final int PRIME = 0x1000193;

    /**
     * Returns a code in [0,2^32-1]
     */
    public static long code(int n) {
        int code = OFFSET_BASIS;

        code ^= 0xff & n;
        code *= PRIME;
        code ^= (0xff00 & n) >>> 8;
        code *= PRIME;
        code ^= (0xff0000 & n) >>> 16;
        code *= PRIME;
        code ^= (0xff000000 & n) >>> 24;
        code *= PRIME;
        return finalizeCode(code);
    }

    /**
     * Returns a code in [0,2^32-1]
     */
    public static long code(String str) {
        int code = OFFSET_BASIS;
        char c;
        int strLength = str.length();
        for (int i = 0; i < strLength; i++) {
            c = str.charAt(i);
            code ^= (0xff & c);
            code *= PRIME;
            code ^= (0xff00 & c) >>> 8;
            code *= PRIME;
        }
        return finalizeCode(code);
    }

    public static int codeAddByte(int code, byte b) {
        int c;
        if (0 <= b) {
            c = b;
        } else {
            c = 0x100 + b;
        }
        code ^= (0xff & c);
        code *= PRIME;
        return code;
    }

    public static int codeAddChar(int code, char c) {
        code ^= (0xff & c);
        code *= PRIME;
        code ^= (0xff00 & c) >>> 8;
        code *= PRIME;
        return code;
    }

    public static int codeAddInt(int code, int c) {
        code ^= (0xff & c);
        code *= PRIME;
        code ^= (0xff00 & c) >>> 8;
        code *= PRIME;
        code ^= (0xff0000 & c) >>> 16;
        code *= PRIME;
        code ^= (0xff000000 & c) >>> 24;
        code *= PRIME;
        return code;
    }

    /**
     * Preserves nonnegative codes.
     * Bijectively maps the negative codes into the interval (Integer.MAX_VALUE, 2^32-1]
     */
    public static long finalizeCode(int code) {
        if (code > 0) {
            return code;
        } else {
            return 0x100000000L - ((long) (code));
        }
    }
}
