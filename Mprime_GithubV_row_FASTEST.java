package com.example.mathematicalconstants;

import java.math.BigInteger;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class Mprime_GithubV_row_FASTEST {
    static AtomicInteger countTotal = new AtomicInteger(0);
    static Queue<BigInteger> unfactoredExponents = new ConcurrentLinkedQueue<>();
    static final BigInteger TWO = BigInteger.valueOf(2);
    static final BigInteger ONE = BigInteger.ONE;

    // Constants
    private static final int OFFSET = 120;
    private static final int ARRAY_SIZE = 666; //574, 584, 594, 658, 584, 600, 698

    private static final int[] BASE_CYCLE_I5_1 = {6, 8, 30, 32, 38, 48, 56, 62, 72, 78, 80, 86, 96, 102, 110, 120};
    private static final int[] BASE_CYCLE_I5_3 = {6, 14, 24, 30, 32, 54, 56, 62, 72, 80, 86, 96, 102, 104, 110, 120};
    private static final int[] BASE_CYCLE_I5_7 = {6, 8, 14, 24, 30, 38, 48, 54, 56, 78, 80, 86, 96, 104, 110, 120};
    private static final int[] BASE_CYCLE_I5_9 = {8, 14, 24, 30, 32, 38, 48, 54, 62, 72, 78, 80, 102, 104, 110, 120};

    private static final int[] BASE_CYCLE_I7_1 = {10, 16, 18, 40, 42, 48, 58, 66, 72, 82, 88, 90, 96, 106, 112, 120};
    private static final int[] BASE_CYCLE_I7_3 = {10, 16, 24, 34, 40, 42, 64, 66, 72, 82, 90, 96, 106, 112, 114, 120};
    private static final int[] BASE_CYCLE_I7_7 = {10, 16, 18, 24, 34, 40, 48, 58, 64, 66, 88, 90, 96, 106, 114, 120};
    private static final int[] BASE_CYCLE_I7_9 = {10, 18, 24, 34, 40, 42, 48, 58, 64, 72, 82, 88, 90, 112, 114, 120};

    private static final int[] BASE_CYCLE_I11_1 = {2, 8, 18, 26, 32, 42, 48, 50, 56, 66, 72, 80, 90, 96, 98, 120};
    private static final int[] BASE_CYCLE_I11_3 = {2, 24, 26, 32, 42, 50, 56, 66, 72, 74, 80, 90, 96, 104, 114, 120};
    private static final int[] BASE_CYCLE_I11_7 = {8, 18, 24, 26, 48, 50, 56, 66, 74, 80, 90, 96, 98, 104, 114, 120};
    private static final int[] BASE_CYCLE_I11_9 = {2, 8, 18, 24, 32, 42, 48, 50, 72, 74, 80, 90, 98, 104, 114, 120};

    private static final int[] BASE_CYCLE_I13_1 = {6, 16, 22, 30, 40, 46, 48, 70, 72, 78, 88, 96, 102, 112, 118, 120};
    private static final int[] BASE_CYCLE_I13_3 = {6, 16, 22, 24, 30, 40, 46, 54, 64, 70, 72, 94, 96, 102, 112, 120};
    private static final int[] BASE_CYCLE_I13_7 = {6, 16, 24, 30, 40, 46, 48, 54, 64, 70, 78, 88, 94, 96, 118, 120};
    private static final int[] BASE_CYCLE_I13_9 = {22, 24, 30, 40, 48, 54, 64, 70, 72, 78, 88, 94, 102, 112, 118, 120};

    // Base Cycles
    private static final Map<Integer, int[][]> BASE_CYCLES = Map.of(
            5, new int[][]{BASE_CYCLE_I5_1, BASE_CYCLE_I5_3, BASE_CYCLE_I5_7, BASE_CYCLE_I5_9},
            7, new int[][]{BASE_CYCLE_I7_1, BASE_CYCLE_I7_3, BASE_CYCLE_I7_7, BASE_CYCLE_I7_9},
            11, new int[][]{BASE_CYCLE_I11_1, BASE_CYCLE_I11_3, BASE_CYCLE_I11_7, BASE_CYCLE_I11_9},
            13, new int[][]{BASE_CYCLE_I13_1, BASE_CYCLE_I13_3, BASE_CYCLE_I13_7, BASE_CYCLE_I13_9}
    );

    public static class ExponentData {
        BigInteger p;
        int pInt;

        public ExponentData(int p) {
            this.p = BigInteger.valueOf(p);
            this.pInt = p;
        }
    }

    public static void main(String[] args) {
        long startTime = System.nanoTime();

        // Prepare exponent data
        List<ExponentData> allExponents = new ArrayList<>();
        for (int start : List.of(5, 7, 11, 13)) {
            allExponents.addAll(generateExponentDataList(1000, start));
        }

        AtomicInteger countFactor = new AtomicInteger(0);

        long lowFactorTimeStart = System.nanoTime();

        // Parallel processing
        allExponents.parallelStream().forEach(data -> {
            int base = data.pInt % 10;
            int origin = getOriginFromStart(data.pInt);
            int[] baseCycle = getBaseCycle(origin, base);
            BigInteger twoToPMinus1 = TWO.pow(data.pInt).subtract(ONE);

            if (checkFactors(data, baseCycle, twoToPMinus1)) {
                countFactor.incrementAndGet();
            } else {
                unfactoredExponents.add(data.p);
            }
        });

        long lowFactorTimeEnd = System.nanoTime();
        System.out.println("Total with at least one 'low' factor: " + countFactor + "/" + countTotal);
        System.out.println("Low factor search runtime (ms): " + (lowFactorTimeEnd - lowFactorTimeStart) / 1_000_000);

        List<BigInteger> unfactored = List.copyOf(unfactoredExponents);

        // 1. Split the unfactored list into two streams
        List<BigInteger> smallExponents = unfactored.parallelStream()
                .filter(p -> p.intValue() <= 16383)
                .collect(Collectors.toList());
        List<BigInteger> largeExponents = unfactored.parallelStream()
                .filter(p -> p.intValue() > 16383)
                .collect(Collectors.toList());

        List<BigInteger> primesFromSmall = filterPrimesSmall(smallExponents);
        List<BigInteger> primesFromLarge = filterPrimesLarge(largeExponents);

        // 3. Combine the results
        List<BigInteger> primeNumberList = new ArrayList<>();
        primeNumberList.addAll(primesFromSmall);
        primeNumberList.addAll(primesFromLarge);

        System.out.println("\nPrime Number List:");
        printList(primeNumberList);

        long totalTime = System.nanoTime() - startTime;
        System.out.println("Total runtime (ms): " + totalTime / 1_000_000);
    }

    private static boolean checkFactors(ExponentData data, int[] baseCycle, BigInteger twoToPMinus1) {
        BigInteger p = data.p;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            BigInteger base = p.multiply(BigInteger.valueOf((long) i * OFFSET)).add(ONE);
            for (int offset : baseCycle) {
                BigInteger test = base.add(p.multiply(BigInteger.valueOf(offset)));
                if (isFactorable(test, twoToPMinus1)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static int getOriginFromStart(int p) {
        return (p - 5) % 12 + 5;
    }

    private static int[] getBaseCycle(int origin, int lastDigit) {
        int idx = switch (lastDigit) {
            case 1 -> 0;
            case 3 -> 1;
            case 7 -> 2;
            default -> 3; // covers 9
        };
        return BASE_CYCLES.get(origin)[idx];
    }

    private static List<ExponentData> generateExponentDataList(int terms, int start) {
        List<ExponentData> list = new ArrayList<>();
        for (int i = 0; i < terms; i++) {
            int exponent = start + 12 * i;
            if (isPrime(exponent)) {
                list.add(new ExponentData(exponent));
                countTotal.incrementAndGet();
            }
        }
        return list;
    }

    private static boolean isFactorable(BigInteger value, BigInteger target) {
        int pInt = target.bitLength();
        return TWO.modPow(BigInteger.valueOf(pInt), value).equals(ONE) && !value.equals(target);
    }

    private static boolean isPrime(int n) {
        int sqrt = (int) Math.sqrt(n);
        for (int i = 3; i <= sqrt; i += 2)
            if (n % i == 0) return false;
        return true;
    }

    // Method 1: Only for Miller-Rabin (for exponents <= 16383)
    private static List<BigInteger> filterPrimesSmall(List<BigInteger> smallExponents) {
        return smallExponents.parallelStream()
                .filter(p -> millerRabinCustom(p))
                .peek(System.out::println)
                .collect(Collectors.toList());
    }

    // Method 2: Lucas-Lehmer
    private static List<BigInteger> filterPrimesLarge(List<BigInteger> largeExponents) {
        return largeExponents.parallelStream()
                .filter(Mprime_GithubV_row_FASTEST::lucasLehmerTest)
                .peek(System.out::println)
                .collect(Collectors.toList());
    }

    private static boolean millerRabinCustom(BigInteger exponent) {
        BigInteger value = ONE.shiftLeft(exponent.intValue()).subtract(ONE);
        BigInteger d = value.subtract(ONE);
        int s = 0;
        while (d.mod(TWO).equals(BigInteger.ZERO)) {
            d = d.shiftRight(1);
            s++;
        }
        BigInteger a = BigInteger.valueOf(3);
        BigInteger x = a.modPow(d, value);
        if (x.equals(ONE) || x.equals(value.subtract(ONE))) return true;

        for (int r = 1; r < s; r++) {
            x = x.modPow(TWO, value);
            if (x.equals(value.subtract(ONE))) return true;
        }
        return false;
    }

    private static boolean lucasLehmerTest(BigInteger p) {
        int intP = p.intValue();
        BigInteger M = ONE.shiftLeft(intP).subtract(ONE);
        BigInteger s = BigInteger.valueOf(4);
        for (int i = 0; i < intP - 2; i++) {
            s = s.multiply(s).subtract(BigInteger.TWO).mod(M);
        }
        return s.equals(BigInteger.ZERO);
    }

    private static void printList(List<BigInteger> list) {
        Collections.sort(list);
        for (int i = 0; i < list.size(); i++) {
            BigInteger p = list.get(i);
            BigInteger mersenneNumber = BigInteger.ONE.shiftLeft(p.intValue()).subtract(BigInteger.ONE);
            System.out.println("Term " + (i + 1) + ": M_" + p + " = " + mersenneNumber);
        }
    }
}
