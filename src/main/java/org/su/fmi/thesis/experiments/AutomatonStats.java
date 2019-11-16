package org.su.fmi.thesis.experiments;

import org.su.fmi.thesis.automaton.Automaton;
import org.su.fmi.thesis.automaton.utils.IntSequence;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class AutomatonStats {

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.out.println("Usage: <input-dir> <output-dir>");
            System.exit(1);
        }

        File inputDir = new File(args[0]);
        File outputDir = new File(args[1]);

        Automaton a = new Automaton();

        int lineNumber = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(new File(inputDir, "automaton.txt")))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (lineNumber == 0) { // initial state is on the first line
                    a.addInitialState(Integer.parseInt(line));
                } else if (lineNumber == 1) { // final states are on the second line separated with spaces
                    String[] finalStates = line.split("\\s");
                    for (String fs : finalStates) {
                        int q = Integer.parseInt(fs);
                        a.addState(q);
                        a.setStateFinality(q, 1);
                    }
                } else { // transitions
                    String[] split = line.split("\\s");
                    int q1 = Integer.parseInt(split[0]);
                    int c = Integer.parseInt(split[1]);
                    int q2 = Integer.parseInt(split[2]);
                    a.addTransition(q1, c, q2);
                }
                ++lineNumber;
            }
        }

        a.sort();

        System.out.println("Number of transitions " + a.getNumberOfTransitions());
        System.out.println("Number of states / reachable / co-reachable " +
                a.getNumberOfStates() + " / " + a.numberOfReachableStates() + " / " + a.numberOfCoReachableStates()
        );
        System.out.println("Number of final states " + a.getNumberOfFinalStates());
        System.out.println();

        a = a.determinize();
        System.out.println("Number of transitions after determinization " + a.getNumberOfTransitions());
        System.out.println("Number of states / reachable / co-reachable after determinization" +
                a.getNumberOfStates() + " / " + a.numberOfReachableStates() + " / " + a.numberOfCoReachableStates()
        );
        System.out.println("Number of final states after determinization " + a.getNumberOfFinalStates());
        System.out.println();

        a = a.minimize();
        System.out.println("Number of transitions after minimization " + a.getNumberOfTransitions());
        System.out.println("Number of states / reachable / co-reachable after minimization" +
                a.getNumberOfStates() + " / " + a.numberOfReachableStates() + " / " + a.numberOfCoReachableStates()
        );
        System.out.println("Number of final states after minimization " + a.getNumberOfFinalStates());
        if (a.containsCycle()) {
            System.out.println("Automaton has cycle");
        } else {
            System.out.println("Automaton has cycle");
        }
        System.out.println();
        a.write(new DataOutputStream(new FileOutputStream(new File(outputDir, "automaton.dat"))));

        Map<Character, Integer> alphabet = readAlphabet(new File(inputDir, "alphabet.tsv"));
        accuracy(a, alphabet, inputDir, outputDir, "train");
        accuracy(a, alphabet, inputDir, outputDir, "dev");
        accuracy(a, alphabet, inputDir, outputDir, "test");
    }

    private static Map<Character, Integer> readAlphabet(File f) throws IOException {
        Map<Character, Integer> alphabet = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] split = line.split("\t", -1);
                if (split.length != 2) {
                    System.out.println("Invalid alphabet format");
                    System.exit(1);
                }
                alphabet.put(split[0].charAt(0), Integer.parseInt(split[1]));
            }
        }
        return alphabet;
    }

    private static void accuracy(
            Automaton a, Map<Character, Integer> alphabet, File inputDir, File outputDir, String dataSetName
    ) throws IOException {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        try (
                BufferedReader br = new BufferedReader(new FileReader(new File(inputDir, dataSetName + ".tsv")));
                BufferedWriter tpw = new BufferedWriter(new FileWriter(new File(outputDir, dataSetName + ".tp.txt")));
                BufferedWriter tnw = new BufferedWriter(new FileWriter(new File(outputDir, dataSetName + ".tn.txt")));
                BufferedWriter fpw = new BufferedWriter(new FileWriter(new File(outputDir, dataSetName + ".fp.txt")));
                BufferedWriter fnw = new BufferedWriter(new FileWriter(new File(outputDir, dataSetName + ".fn.txt")))
        ) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] split = line.split("\t");
                if (split.length != 2) {
                    System.out.println("Invalid format");
                    System.exit(1);
                }
                int label = Integer.parseInt(split[0]);
                String word = split[1];
                IntSequence seq = new IntSequence(word.length(), -1);
                for (int i = 0; i < word.length(); ++i) {
                    seq.add(alphabet.get(word.charAt(i)));
                }
                int prediction = a.accepts(seq);

                if (label == 1) {
                    if (prediction == 1) {
                        ++tp;
                        tpw.write(word);
                        tpw.newLine();
                    } else {
                        ++fn;
                        fnw.write(word);
                        fnw.newLine();
                    }
                } else {
                    if (prediction == 1) {
                        ++fp;
                        fpw.write(word);
                        fpw.newLine();
                    } else {
                        tn++;
                        tnw.write(word);
                        tnw.newLine();
                    }
                }
            }
        }

        System.out.println(String.format("%s : TP : %s TN : %s FP : %s FN : %s ACC : %s",
                dataSetName, tp, tn, fp, fn, ((tp + tn) / (double) (tp + tn + fp + fn))
        ));
    }
}
