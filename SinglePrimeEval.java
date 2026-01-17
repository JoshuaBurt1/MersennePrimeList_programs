package com.example.mathematicalconstants;

import java.math.BigInteger;

// Lucas-Lehmer primality test is faster for p > (16383 = 2^14-1) -> faster for n > 2^(2^14-1)-1
// Miller-Rabin primality test is faster for p <= (16383 = 2^14-1) -> faster for n < 2^(2^14-1)-1
//TODO: Why.

//Lucas-Lehmer                      (L: 30, L: 30 A. Greek)
//11111 : 812 ms    slower
//15553 : 1931 ms   slower
//16291 : 2101 ms   slower
//16383 : 2199 ms   slower
//16384             slower

//16385             faster
//16386             faster
//16387 : 2226 ms   faster
//16391 : 2123 ms   faster
//16431 : 2219 ms   faster
//16531 : 2296 ms   faster
//16481 : 2188 ms   faster
//17001 : 2402 ms   faster
//17991 : 2862 ms   faster
//21111 : 4412 ms   faster
//23331 : 5429 ms   faster
//25791 : 6885 ms   faster
//34561 : 14309 ms  faster
//Miller-Rabin
//11111 : 590 ms      10101101100111
//15553 : 1547 ms     11110011000001
//16291 : 1711 ms     11111110100011
//16383 : 1755 ms     11111111111111 (14 digits - unusually faster up to 14 digits "14:Miller -> L:30,L:30,M:40,R:100 A. Greek -> 34.13 1 sigma 1 side)
//16387 : 2716 ms
//16391 : 2633 ms
//16431 : 2656 ms
//16481 : 2692 ms
//16531 : 2729 ms
//17001 : 2916 ms    100001001101001
//17991 : 3533 ms    100011001000111
//21111 : 5984 ms    101001001110111
//23331 : 7791 ms    101101100100011
//25791 : 10379 ms   110010010111111
//34561 : 23931 ms  1000011100000001

//For single prime evaluation set the exponent value and exponent value range to the same value
public class SinglePrimeEval {
    static final BigInteger TWO = BigInteger.valueOf(2);
    static final BigInteger ONE = BigInteger.ONE;
    public static void main(String[] args) {
        long totalStartTimeB = System.nanoTime();

        for (int exponentValue = 16383; exponentValue <= 16383; exponentValue++) {

            BigInteger exponent = BigInteger.valueOf(exponentValue);
            BigInteger two = BigInteger.valueOf(3);
            BigInteger mersenneNumber = two.pow(exponent.intValue()).subtract(BigInteger.ONE);

            // Measure execution time for Lucas-Lehmer test
            long startTimeLucas1 = System.nanoTime();
            if(!lucasLehmerOriginal(exponent)) {
                System.out.println("Composite");
            }
            long endTimeLucas1 = System.nanoTime();
            long durationLucas1 = (endTimeLucas1 - startTimeLucas1) / 1_000_000; // Convert nanoseconds to milliseconds
            System.out.println("Execution time (Lucas-Lehmer original): " + durationLucas1 + " milliseconds");

            // Measure execution time for Lucas-Lehmer test
            long startTimeLucas = System.nanoTime();
            if(!lucasLehmerTest(exponent)) {
                System.out.println("Composite");
            }
            long endTimeLucas = System.nanoTime();
            long durationLucas = (endTimeLucas - startTimeLucas) / 1_000_000; // Convert nanoseconds to milliseconds
            System.out.println("Execution time (Lucas-Lehmer test): " + durationLucas + " milliseconds");


            // Measure execution time for BigInteger.isProbablePrime
            long startTimeC = System.nanoTime();
            if(!probablePrime(exponentValue)) {
                System.out.println("Composite");
            }
            long endTimeC = System.nanoTime();
            long durationC = (endTimeC - startTimeC) / 1_000_000; // Convert nanoseconds to milliseconds
            System.out.println("Execution time (BigInteger.isProbablePrime): " + durationC + " milliseconds");


            // Measure execution time for custom Miller-Rabin with base 3
            long startTimeB = System.nanoTime();
            if(!millerRabinCustom(BigInteger.valueOf(exponentValue))) {
                System.out.println("Composite");
            }
            long endTimeB = System.nanoTime();
            long durationB = (endTimeB - startTimeB) / 1_000_000; // Convert nanoseconds to milliseconds
            System.out.println("Execution time (Custom Miller-Rabin with base 3): " + durationB + " milliseconds");


        }
        long totalEndTimeB = System.nanoTime();
        long totalDuration = (totalEndTimeB - totalStartTimeB) / 1_000_000; // Convert nanoseconds to milliseconds
        System.out.println("Total time: " + totalDuration + " milliseconds");

    }

    // Uses Miller-Rabin primality testing (BigInteger default method)
    private static boolean probablePrime(int value) {
        // Check if a solution exists for formula1 only
        BigInteger mp = BigInteger.TWO.pow(value).subtract(BigInteger.ONE);
        if(mp.isProbablePrime(10)){
            System.out.println("The number is a prime: " + value);
            return true;
        }
        System.out.println("The number is not a prime: " + value);

        return false;
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

    private static boolean lucasLehmerTest(BigInteger exponent) {
        //if (exponent.equals(BigInteger.ONE)) {
        //    System.out.println("The number is composite: " + (BigInteger.valueOf(2).pow(1).subtract(BigInteger.ONE)));
        //    return false;  // 2^1 - 1 = 1, not prime
        //}

        // Mersenne number: M_p = 2^p - 1
        BigInteger mersenneNumber = BigInteger.ONE.shiftLeft(exponent.intValue()).subtract(BigInteger.ONE);

        // Initial value for S_0
        BigInteger s = BigInteger.valueOf(4);
        BigInteger two = BigInteger.valueOf(2);

        // Apply recurrence relation S_n = (S_{n-1}^2 - 2) % M_p
        for (BigInteger n = BigInteger.ONE; n.compareTo(exponent.subtract(BigInteger.ONE)) < 0; n = n.add(BigInteger.ONE)) {
            s = s.multiply(s).subtract(two); // S_{n-1}^2 - 2
            s = s.mod(mersenneNumber); // Modulo M_p
        }

        // Check if S_{p-2} % M_p == 0
        if (s.equals(BigInteger.ZERO)) {
            System.out.println("The number is a Mersenne prime: " + mersenneNumber);
            return true;
        } else {
            System.out.println("The number is composite: " + mersenneNumber);
            return false;
        }
    }

    // Lucas-Lehmer Primality test for Mersenne numbers
    private static boolean lucasLehmerOriginal(BigInteger exponent) {
        // Mersenne number: M_p = 2^p - 1
        BigInteger two = BigInteger.valueOf(2);
        BigInteger mersenneNumber = two.pow(exponent.intValue()).subtract(BigInteger.ONE);

        // Initial value for S_0
        BigInteger s = BigInteger.valueOf(4);

        // Apply recurrence relation S_n = (S_{n-1}^2 - 2) % M_p
        for (BigInteger n = BigInteger.ONE; n.compareTo(exponent.subtract(BigInteger.ONE)) < 0; n = n.add(BigInteger.ONE)) {
            s = s.multiply(s).subtract(BigInteger.valueOf(2)); // S_{n-1}^2 - 2
            s = s.mod(mersenneNumber); // Modulo M_p
        }

        // Check if S_{p-2} % M_p == 0
        if (s.equals(BigInteger.ZERO)) {
            System.out.println("The number is a Mersenne prime: " + mersenneNumber);
            return true;
        } else {
            System.out.println("The number is composite: " + mersenneNumber);
            return false;
        }
    }
}

//27th Mp : 2^44497-1 = 335850 ms / 5.6 min //value.isProbablePrime(10); hardware: i7
//27th Mp : 2^44497-1 = 166667 ms / 2.77 min //value.isProbablePrime(10); hardware: i9
//27th Mp : 2^44497-1 = 49714 ms / 0.82 min; //Custom Miller-Rabin Primality test base 3; hardware: i9
//27th Mp : 2^44497-1 = 26981 ms / 0.449 min; //Lucas-Lehmer Primality test; hardware: i9

//28th Mp : 2^86243-1 = 2410454 ms or 40.17 min //value.isProbablePrime(10); hardware: i7
//28th Mp : 2^86243-1 = 1113787 ms or 18.56 min //value.isProbablePrime(10); hardware: i9
//28th Mp : 2^86243-1 = 390527 ms or 6.51 min //Custom Miller-Rabin Primality test base 3; hardware: i9
//28th Mp : 2^86243-1 = 183914 ms / 3.06 min; //Lucas-Lehmer Primality test; hardware: i9

//29th Mp : 2^110503-1 = 4418631 ms or 73.6 min (certainty: 10); hardware: i7
//29th Mp : 2^110503-1 =
//29th Mp : 2^110503-1 =                      //Custom Miller-Rabin Primality test base 3; hardware: i9
//29th Mp : 2^110503-1 = 279522 ms / 4.65 min; //Lucas-Lehmer Primality test; hardware: i9
//29th Mp : 2^110503-1 =

//30th Mp : 2^132049-1 = 7401334 ms or 123.4 min (certainty: 10);
//31st Mp : 2^216091-1 = 31820747 ms or 530.3 min (certainty: 10);

