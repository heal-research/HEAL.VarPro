using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace HEAL.VarPro.Test {
  [TestClass]
  public class ProblemsTest {

    [TestMethod]
    public void SingleNonLinearTerm() {
      var N = 20;
      void phiFunc(double[] alpha, ref double[,] phi) {
        phi = new double[N, 1];
        var t = 0.0;
        var deltaT = 0.05;
        for (int i = 0; i < N; i++) {
          phi[i, 0] = Math.Cos(-alpha[0] * t);
          t += deltaT;
        }
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        ind = new int[,] {
          { 0 }, // index of term
          { 0 } // index of alpha
        };
        Jac = new double[N, ind.GetLength(1)];

        var t = 0.0;
        var deltaT = 0.05;
        for (int i = 0; i < N; i++) {
          Jac[i, 0] = -t * Math.Sin(alpha[0] * t);
          t += deltaT;
        }
      }

      var random = new Random(1234);
      var c = new double[] { 1 }; // true c
      var optAlpha = new double[] { 3 }; // true alpha
      double[,] phi = null;
      phiFunc(optAlpha, ref phi);

      var y = new double[N];
      var yTest = new double[N];
      var noiseSigma = 0.0; // no noise for the test
      for (int i = 0; i < N; i++) {
        yTest[i] = c[0] * phi[i, 0];
        y[i] = yTest[i] + RandNormal(random, noiseSigma);
      }

      double[] initialAlpha = new[] { 0.1 };
      TestVarPro(initialAlpha, y, phiFunc, jacFunc, phiFunc, yTest, out _, useWGCV: false);
    }


    [TestMethod]
    public void ChenLakeErie() {
      // Guang-Yong Chen, Min Gan, C.L. Philip Chen, Han-Xiong Li, 
      // A Regularized Variable Projection Algorithm for Separable 
      // Nonlinear Least-Squares Problems
      // IEEE Transactions on Automatic Control, Vol. 64, No. 2,
      // Feb. 2019

      // ExpVar model for Lake Erie data in III.B

      var N = 300; // first 300 data points used for training
      void phiFunc(double[] alpha, ref double[,] phi) {
        // ExpAR model suggested by Teräsvirta
        var y = MonthlyLakeErieLevels_1921_1970.Take(N).ToArray();
        var p = 11;
        var z = alpha[0];
        var g = alpha[1];
        phi = new double[N - p, 24];
        for (int t = p; t < N; t++) {
          var colIdx = 0;
          phi[t - p, colIdx++] = 1.0;
          for (int j = 1; j <= p; j++) {
            phi[t - p, colIdx++] = y[t - j];
          }
          phi[t - p, colIdx++] = Math.Exp(-g * (y[t - 1] - z) * (y[t - 1] - z));
          for (int j = 1; j <= p; j++) {
            phi[t - p, colIdx++] = Math.Exp(-g * (y[t - 1] - z) * (y[t - 1] - z)) * y[t - j];
          }
        }
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        ind = new int[,] {
        { 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23 }, // index of term
        { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 } // index of alpha
      };
        var p = 11;
        var z = alpha[0];
        var g = alpha[1];
        Jac = new double[N - p, ind.GetLength(1)];
        var y = MonthlyLakeErieLevels_1921_1970.Take(N).ToArray();


        for (int t = p; t < N; t++) {
          var colIdx = 0;

          Jac[t - p, colIdx++] = 2 * g * (y[t - 1] - z) * Math.Exp(-g * (y[t - 1] - z) * (y[t - 1] - z));
          Jac[t - p, colIdx++] = (y[t - 1] - z) * (y[t - 1] - z) * (-Math.Exp(-g * (y[t - 1] - z) * (y[t - 1] - z)));

          for (int j = 1; j <= p; j++) {
            Jac[t - p, colIdx++] = 2 * g * (y[t - 1] - z) * Math.Exp(-g * (y[t - 1] - z) * (y[t - 1] - z)) * y[t - j];
            Jac[t - p, colIdx++] = (y[t - 1] - z) * (y[t - 1] - z) * (-Math.Exp(-g * (y[t - 1] - z) * (y[t - 1] - z))) * y[t - j];
          }
        }
      }

      var y = MonthlyLakeErieLevels_1921_1970.Take(N).Skip(11).ToArray(); // predict y(t+1)


      void phiFuncTest(double[] alpha, ref double[,] phi) {
        var p = 11;
        var y = MonthlyLakeErieLevels_1921_1970.Skip(N - p).ToArray();
        var z = alpha[0];
        var g = alpha[1];
        phi = new double[N, 24];
        for (int t = p; t < N + p; t++) {
          var colIdx = 0;
          phi[t - p, colIdx++] = 1.0;
          for (int j = 1; j <= p; j++) {
            phi[t - p, colIdx++] = y[t - j];
          }
          phi[t - p, colIdx++] = Math.Exp(-g * (y[t - 1] - z) * (y[t - 1] - z));
          for (int j = 1; j <= p; j++) {
            phi[t - p, colIdx++] = Math.Exp(-g * (y[t - 1] - z) * (y[t - 1] - z)) * y[t - j];
          }
        }
      }
      var yTest = MonthlyLakeErieLevels_1921_1970.Skip(N).ToArray();

      double[] initialAlpha = new[] { 15.0, 5 }; // not given in the paper
      var alpha = (double[])initialAlpha.Clone();
      Console.WriteLine("O'Leary & Rust (without regularization):");
      TestVarPro(alpha, y, phiFunc, jacFunc, phiFuncTest, yTest, out var yPredClassical, useWGCV: false);
      {
        // compare to data from paper
        Console.WriteLine();
        Console.WriteLine("Reported by Chen et al. for 'classical VP'");
        Console.WriteLine($"MSE (training)    0.1819");
        Console.WriteLine($"MSE (test)      206.9916");
      }

      Console.WriteLine();
      Console.WriteLine("Chen et al. (with WGCV):");
      alpha = (double[])initialAlpha.Clone();
      TestVarPro(alpha, y, phiFunc, jacFunc, phiFuncTest, yTest, out var yPredRegularized, useWGCV: true);
      var alphaRegularized = alpha;
      {
        // compare to data from paper
        Console.WriteLine();
        Console.WriteLine("Reported by Chen et al.: w: 0.303, lambda: 0.623");
        Console.WriteLine($"MSE (training)    0.1881");
        Console.WriteLine($"MSE (test)        0.2617");
      }

      // generate data for figure
      Console.WriteLine("Index\tMeasured\tClassical VP\tRegularized VP");
      for (int i = 0; i < yTest.Length; i++) {
        Console.WriteLine($"{i + 1}\t{yTest[i]}\t{yPredClassical[i]}\t{yPredRegularized[i]}");
      }
    }

    [TestMethod]
    public void ChenExponential() {
      // Guang-Yong Chen, Min Gan, C.L. Philip Chen, Han-Xiong Li, 
      // A Regularized Variable Projection Algorithm for Separable 
      // Nonlinear Least-Squares Problems
      // IEEE Transactions on Automatic Control, Vol. 64, No. 2,
      // Feb. 2019

      // III.C Parameter Estimation of a Complex Exponential Model

      var N = 151;

      void phiFunc(double[] alpha, ref double[,] phi) {
        phi = new double[N, 2];
        var t = 0.0;
        var deltaT = 0.01;
        for (int i = 0; i < N; i++) {
          phi[i, 0] = Math.Exp(-alpha[1] * t) * Math.Cos(alpha[2] * t);
          phi[i, 1] = Math.Exp(-alpha[0] * t) * Math.Cos(alpha[1] * t);
          t += deltaT;
        }
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        ind = new int[,] {
          { 0, 0, 1, 1 }, // index of term
          { 1, 2, 0, 1 } // index of alpha
         };
        Jac = new double[N, ind.GetLength(1)];

        var t = 0.0;
        var deltaT = 0.01;
        for (int i = 0; i < N; i++) {
          Jac[i, 0] = -t * Math.Exp(-alpha[1] * t) * Math.Cos(alpha[2] * t);
          Jac[i, 1] = -t * Math.Exp(-alpha[1] * t) * Math.Sin(alpha[2] * t);

          Jac[i, 2] = -t * Math.Exp(-alpha[0] * t) * Math.Cos(alpha[1] * t);
          Jac[i, 3] = -t * Math.Exp(-alpha[0] * t) * Math.Sin(alpha[1] * t);

          t += deltaT;
        }
      }

      var y = new double[N];
      var yTest = new double[N]; // y = yTest + noise

      var optAlpha = new double[] { 1, 1, 15 }; // true alpha
      var c = new double[] { 3, 5 }; // true c
      var noiseSigma = 0.4;
      double[,] phi = null;
      phiFunc(optAlpha, ref phi);

      var random = new Random(123);
      for (int i = 0; i < N; i++) {
        yTest[i] = c[0] * phi[i, 0]
               + c[1] * phi[i, 1];
        y[i] = yTest[i] + RandNormal(random, noiseSigma);
      }

      double[] initialAlpha = new[] { 0.8457, 2.3331, 7.4757 };
      Console.WriteLine("O'Leary & Rust (without regularization):");
      Console.WriteLine("----------------------------------------");
      var alpha = (double[])initialAlpha.Clone();
      TestVarPro(alpha, y, phiFunc, jacFunc, phiFunc, yTest, out _, useWGCV: false);
      Console.WriteLine($"Optimal    alpha: {string.Join(" ", optAlpha.Select(ai => ai.ToString("e3")))}");
      Console.WriteLine($"Identified alpha: {string.Join(" ", alpha.Select(ai => ai.ToString("e3")))}");
      {
        // compare to data from paper
        Console.WriteLine();
        Console.WriteLine("Reported by Chen et al. for 'classical VP'");
        alpha = new[] { 3.23, 2.73, 3.28 };
        phiFunc(alpha, ref phi);
        var coeff = new[] { -40.53, 48.65 };
        var yPred = new double[y.Length];
        var r = new double[y.Length];
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < coeff.Length; j++) {
            yPred[i] += phi[i, j] * coeff[j];
          }
          r[i] = yTest[i] - yPred[i];
        }
        Console.WriteLine($"||coeff||²        {Dot(coeff, coeff):e4}");
        Console.WriteLine($"||resid_test||²   {Dot(r, r):e4}");
        Console.WriteLine($"||resid_test||    {Math.Sqrt(Dot(r, r)):e4}");
        Console.WriteLine($"MSE (test)        {Dot(r, r) / r.Length:e4}");
      }



      Console.WriteLine();
      Console.WriteLine("Chen et. al. (with WGCV):");
      Console.WriteLine("----------------------------------------");
      alpha = (double[])initialAlpha.Clone();
      TestVarPro(alpha, y, phiFunc, jacFunc, phiFunc, yTest, out _, useWGCV: true);

      Console.WriteLine($"Optimal    alpha: {string.Join(" ", optAlpha.Select(ai => ai.ToString("e3")))}");
      Console.WriteLine($"Identified alpha: {string.Join(" ", alpha.Select(ai => ai.ToString("e3")))}");

      {
        // compare to data from paper
        Console.WriteLine();
        Console.WriteLine("Reported by Chen et al.: w: 0.95, lambda: 0.098");
        alpha = new[] { 1.06, 1.01, 14.98 };
        phiFunc(alpha, ref phi);
        var coeff = new[] { 2.99, 5.06 };
        var yPred = new double[y.Length];
        var r = new double[y.Length];
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < coeff.Length; j++) {
            yPred[i] += phi[i, j] * coeff[j];
          }
          r[i] = yTest[i] - yPred[i];
        }
        Console.WriteLine($"||coeff||²        {Dot(coeff, coeff):e4}");
        Console.WriteLine($"||resid_test||²   {Dot(r, r):e4}");
        Console.WriteLine($"||resid_test||    {Math.Sqrt(Dot(r, r)):e4}");
        Console.WriteLine($"MSE (test)        {Dot(r, r) / r.Length:e4}");
      }
    }

    [TestMethod]
    public void ChenTableIII() {
      // Guang-Yong Chen, Min Gan, C.L. Philip Chen, Han-Xiong Li, 
      // A Regularized Variable Projection Algorithm for Separable 
      // Nonlinear Least-Squares Problems
      // IEEE Transactions on Automatic Control, Vol. 64, No. 2,
      // Feb. 2019

      // III.C Parameter Estimation of a Complex Exponential Model
      // Noise sensitivity

      var N = 151;

      void phiFunc(double[] alpha, ref double[,] phi) {
        phi = new double[N, 2];
        var t = 0.0;
        var deltaT = 0.01;
        for (int i = 0; i < N; i++) {
          phi[i, 0] = Math.Exp(-alpha[1] * t) * Math.Cos(alpha[2] * t);
          phi[i, 1] = Math.Exp(-alpha[0] * t) * Math.Cos(alpha[1] * t);
          t += deltaT;
        }
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        ind = new int[,] {
          { 0, 0, 1, 1 }, // index of term
          { 1, 2, 0, 1 } // index of alpha
         };
        Jac = new double[N, ind.GetLength(1)];

        var t = 0.0;
        var deltaT = 0.01;
        for (int i = 0; i < N; i++) {
          Jac[i, 0] = -t * Math.Exp(-alpha[1] * t) * Math.Cos(alpha[2] * t);
          Jac[i, 1] = -t * Math.Exp(-alpha[1] * t) * Math.Sin(alpha[2] * t);

          Jac[i, 2] = -t * Math.Exp(-alpha[0] * t) * Math.Cos(alpha[1] * t);
          Jac[i, 3] = -t * Math.Exp(-alpha[0] * t) * Math.Sin(alpha[1] * t);

          t += deltaT;
        }
      }

      var y = new double[N];
      var yTest = new double[N]; // y = yTest + noise

      var optAlpha = new double[] { 1, 1, 15 }; // true alpha
      var c = new double[] { 3, 5 }; // true c
      double[,] phi = null;
      phiFunc(optAlpha, ref phi);

      for (int i = 0; i < N; i++) {
        yTest[i] = c[0] * phi[i, 0]
                 + c[1] * phi[i, 1];
      }


      double[] initialAlpha = new[] { 0.8457, 2.3331, 7.4757 };

      var random = new Random(123);
      Console.WriteLine();
      Console.WriteLine("noise  classical VP    regularized VP");
      foreach (var noiseSigma in new[] { 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }) {


        // average of 30 reps
        int reps = 30;
        var mseClassical = new List<double>();
        var mseRegularized = new List<double>();
        for (int rep = 0; rep < reps; rep++) {
          for (int i = 0; i < N; i++) {
            y[i] = yTest[i] + RandNormal(random, noiseSigma);
          }

          var alpha = (double[])initialAlpha.Clone();
          VariableProjection.Fit(phiFunc, jacFunc, y, alpha, out var coeff, out _, useWGCV: false, eps: 1e-12);
          phiFunc(alpha, ref phi);

          var yPred = new double[y.Length];
          var r = new double[y.Length];
          for (int i = 0; i < N; i++) {
            for (int j = 0; j < coeff.Length; j++) {
              yPred[i] += phi[i, j] * coeff[j];
            }
            r[i] = yTest[i] - yPred[i];
          }
          mseClassical.Add(Dot(r, r) / r.Length);

          alpha = (double[])initialAlpha.Clone();
          VariableProjection.Fit(phiFunc, jacFunc, y, alpha, out coeff, out _, useWGCV: true, eps: 1e-12);
          phiFunc(alpha, ref phi);

          yPred = new double[y.Length];
          r = new double[y.Length];
          for (int i = 0; i < N; i++) {
            for (int j = 0; j < coeff.Length; j++) {
              yPred[i] += phi[i, j] * coeff[j];
            }
            r[i] = yTest[i] - yPred[i];
          }
          mseRegularized.Add(Dot(r, r) / r.Length);

        }

        Console.WriteLine($"{noiseSigma,5:g2} {mseClassical.Average(),15:e5} {mseRegularized.Average(),15:e5}");
      }
    }

    [TestMethod]
    public void Linear() {
      var N = 30;

      void phiFunc(double[] alpha, ref double[,] phi) {
        var xRand = new Random(1234);
        phi = new double[N, 2];
        for (int i = 0; i < N; i++) {
          phi[i, 0] = xRand.NextDouble();
          phi[i, 1] = xRand.NextDouble();
        }
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        ind = new int[,] {
          { }, // index of term
          { } // index of alpha
        };
        Jac = new double[N, ind.GetLength(1)];
      }

      var c = new double[] { 3, 5 }; // true c
      var noise_range = 0.4;
      var y = new double[N];
      var yTest = new double[N];
      double[,] phi = null;
      phiFunc(new double[0], ref phi);
      var noiseRand = new Random(1234);
      for (int i = 0; i < N; i++) {
        yTest[i] = c[0] * phi[i, 0]
                   + c[1] * phi[i, 1];
        y[i] = yTest[i] + noiseRand.NextDouble() * noise_range - noise_range / 2; // train = test + noise
      }

      double[] initialAlpha = new double[0];
      TestVarPro(initialAlpha, y, phiFunc, jacFunc, phiFunc, yTest, out _, useWGCV: false);
    }

    [TestMethod]
    public void OLearyRustCos() {
      var N = 20;

      void phiFunc(double[] alpha, ref double[,] phi) {
        phi = new double[N, 4];
        var t = 0.0;
        var deltaT = 0.05;
        for (int i = 0; i < N; i++) {
          phi[i, 0] = Math.Cos(-alpha[0] * t);
          phi[i, 1] = Math.Cos(-alpha[1] * t);
          phi[i, 2] = Math.Cos(-alpha[2] * t);
          phi[i, 3] = Math.Cos(-alpha[3] * t);
          t += deltaT;
        }
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        ind = new int[,] {
        { 0, 1, 2, 3 }, // index of term
        { 0, 1, 2, 3 } // index of alpha
       };
        Jac = new double[N, ind.GetLength(1)];

        var t = 0.0;
        var deltaT = 0.05;
        for (int i = 0; i < N; i++) {
          Jac[i, 0] = -t * Math.Sin(alpha[0] * t);
          Jac[i, 1] = -t * Math.Sin(alpha[1] * t);
          Jac[i, 2] = -t * Math.Sin(alpha[2] * t);
          Jac[i, 3] = -t * Math.Sin(alpha[3] * t);
          t += deltaT;
        }
      }

      var optAlpha = new double[] { 3, -4, 6, 1 }; // true alpha
      var c = new double[] { 1, 2, 3, 4 }; // true c
      double[,] optPhi = null;
      phiFunc(optAlpha, ref optPhi);

      var y = new double[N];
      for (int i = 0; i < N; i++) {
        y[i] = c[0] * optPhi[i, 0] +
               c[1] * optPhi[i, 1] +
               c[2] * optPhi[i, 2] +
               c[3] * optPhi[i, 3];
      }

      double[] initialAlpha = new[] { 2.5, -3.5, 5.5, 1.0 };
      TestVarPro(initialAlpha, y, phiFunc, jacFunc, phiFunc, y, out _, useWGCV: false); // train == test
    }

    [TestMethod]
    public void Krogh() {
      // Fred T. Krogh, Efficient Implementation of a Variable Projection Algorithm
      //                for Nonlinear Least Squares Problems, Communications of the
      //                ACM, Numerical Mathematics, Vol 17, No. 3, March 1974,
      //                pp. 167-169

      // Dianne P. O'Leary Bert W. Rust, Variable Projection for Nonlinear Least 
      //                Squares Problems
      // https://www.cs.umd.edu/~oleary/software/varpro/



      var V = new double[] { 0.572, 0.552, 0.533, 0.518, 0.500, 0.487, 0.470, 0.452, 0.437, 0.415, 0.396, 0.373, 0.341, 0.312, 0.266, 0.220, 0.167, 0.103, 0.075 };
      var ID = new double[] { 1.00e-2, 6.25e-3, 3.75e-3, 2.50e-3, 1.50e-3, 1.00e-3, 6.25e-4, 3.75e-4, 2.50e-4, 1.50e-4, 1.00e-4, 6.25e-5, 3.75e-5, 2.50e-5, 1.50e-5, 1.00e-5, 6.25e-6, 3.75e-6, 2.50e-6 };

      var c = new double[] { 7.4334e-08, 2.0094e-11, 3.1559e-5 }; // true c as reported by O'Leary and Rust  6 iterations required to converge
      var N = 19;
      var Q = 39.206807432432432;
      var y = new double[N];
      for (int i = 0; i < N; i++) {
        y[i] = 1.0;
      }

      void phiFunc(double[] alpha, ref double[,] phi) {
        phi = new double[N, 3];
        var t = 0.0;
        var deltaT = 0.05;
        var a1 = alpha[0];
        var a2 = alpha[1];
        var a3 = alpha[2];
        for (int i = 0; i < N; i++) {
          phi[i, 0] = (Math.Exp(Q * (V[i] - ID[i] * a1) * a2) - 1) / ID[i];
          phi[i, 1] = (Math.Exp(Q * (V[i] - ID[i] * a1) * a3) - 1) / ID[i];
          phi[i, 2] = (V[i] - ID[i] * a1) / ID[i];
          t += deltaT;
        }
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        ind = new int[,] {
        { 0, 0, 1, 1, 2 }, // index of term
        { 0, 1, 0, 2, 0 } // index of alpha
        };

        Jac = new double[N, ind.GetLength(1)];

        var a1 = alpha[0];
        var a2 = alpha[1];
        var a3 = alpha[2];
        for (int i = 0; i < N; i++) {
          Jac[i, 0] = -a2 * Q * Math.Exp(a2 * Q * (V[i] - a1 * ID[i]));
          Jac[i, 1] = (Q * (V[i] - a1 * ID[i]) * Math.Exp(a2 * Q * (V[i] - a1 * ID[i]))) / ID[i];
          Jac[i, 2] = -a3 * Q * Math.Exp(a3 * Q * (V[i] - a1 * ID[i]));
          Jac[i, 3] = (Q * (V[i] - a1 * ID[i]) * Math.Exp(a3 * Q * (V[i] - a1 * ID[i]))) / ID[i];
          Jac[i, 4] = -1;
        }
      }

      // start point 1 for Krogh problem
      Console.WriteLine("First starting point { 2.055, 0.4721435316336, 1.0 }:");
      var initialAlpha = new[] { 2.055, 0.4721435316336, 1.0 }; // 4 iterations required to converge

      // var initialAlpha = new[] { 2.055, 0.05, 1.0 }; // second start point. 6 iterations required to converge
      Console.WriteLine("  Classical VP:");
      var alpha = (double[])initialAlpha.Clone();
      TestVarPro(alpha, y, phiFunc, jacFunc, phiFunc, y, out _, useWGCV: false); // train == test

      Console.WriteLine("  WGCV-Regularized VP:");
      alpha = (double[])initialAlpha.Clone();
      TestVarPro(alpha, y, phiFunc, jacFunc, phiFunc, y, out _, useWGCV: true); // train == test

      // var optAlpha = new double[] { 1.672719, 0.4272597, 0.9157357 }; // true alpha
      Console.WriteLine("Second starting point { 2.055, 0.05, 1.0 }:");
      initialAlpha = new[] { 2.055, 0.05, 1.0 }; // 4 iterations required to converge
      Console.WriteLine("  Classical VP:");
      alpha = (double[])initialAlpha.Clone();
      TestVarPro(alpha, y, phiFunc, jacFunc, phiFunc, y, out _, useWGCV: false); // train == test

      Console.WriteLine("  WGCV-Regularized VP:");
      alpha = (double[])initialAlpha.Clone();
      TestVarPro(alpha, y, phiFunc, jacFunc, phiFunc, y, out _, useWGCV: true); // train == test
    }

    [TestMethod]
    public void Poly10_Linear() {
      // y = x1 x2 + x3 x4 + x5 x6 + x1 x7 x9 + x3 x6 x10
      // no noise
      // we use a polynomial feature expansion to a polynomial of total degree 3
      // There are no non-linear features, so we only test the code for least-squares solving.

      var terms = new List<string>();
      var n = 100;
      var nTerms = Binom(11 + 3 - 1, 3);
      var x = new double[n, 11];
      var f = new double[n, 10 * nTerms];
      var i = 0;
      for (int r = 0; r < n; r++) { x[r, i] = 1.0; } // const column

      // 10 random variables ~ N(0.0, 1.0)
      i++;
      var rand = new System.Random(1234);
      for (; i <= 10; i++) {
        for (int r = 0; r < n; r++) { x[r, i] = RandNormal(rand, 1.0); }
      }


      i = 0;
      // generate all polynomial terms
      for (int aIdx = 0; aIdx <= 10; aIdx++) {
        for (int bIdx = aIdx; bIdx <= 10; bIdx++) {
          for (int cIdx = bIdx; cIdx <= 10; cIdx++) {
            var term = "";
            if (aIdx > 0) term += "x" + aIdx;
            if (bIdx > 0) term += "x" + bIdx;
            term += cIdx == 0 ? "1" : ("x" + cIdx);
            terms.Add(term);
            for (int r = 0; r < n; r++) { f[r, i] = x[r, aIdx] * x[r, bIdx] * x[r, cIdx]; }
            i++;
          }
        }
      }

      Debug.Assert(i == nTerms); // all terms generated?

      // target
      var y = new double[n];
      for (int r = 0; r < n; r++)
        y[r] = x[r, 1] * x[r, 2]
          + x[r, 3] * x[r, 4]
          + x[r, 5] * x[r, 6]
          + x[r, 1] * x[r, 7] * x[r, 9]
          + x[r, 3] * x[r, 6] * x[r, 10];



      void phiFunc(double[] alpha, ref double[,] phi) {
        if (phi == null) phi = (double[,])f.Clone();
        else Array.Copy(f, phi, phi.Length);
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        // no non-linear params
        if (Jac == null) {
          Jac = new double[n, 0];
          ind = new int[2, 0];
        }
      }


      VarPro.VariableProjection.Fit(phiFunc, jacFunc, y, new double[0], out var coeff, out var report, useWGCV: true);
      var threshold = 0.2;
      var selectedTerms = 0;
      for (i = 0; i < coeff.Length; i++) {
        if (Math.Abs(coeff[i]) > threshold) {
          if (selectedTerms > 0) Console.Write(" + ");
          Console.Write($"{coeff[i]:e4} {terms[i]}");
          selectedTerms++;
        }
      }
    }

    [TestMethod]
    public void SumOfExponentials() {
      // Predict
      //  y = exp(- (x1 - 1)²) / (1.2 + (x2 - 2.5)²)
      // using a sum of terms of the form exp(a*x)
      #region data
      var xy = new double[,] {
{0.471677717536613, 0.74741918836893, 0.177089728292188},
{2.22447705260117, 2.384135614122, 0.184005316835132},
{2.79197140573337, 2.98098888878734, 0.0281620545293636},
{3.67347702686057, 3.94920519998977, 0.000238423706801869},
{3.59747060185705, 1.30087382465156, 0.000445266297963256},
{1.58490849469903, 1.425988084060, 0.301790501193091},
{2.66258209931259, 0.33589706601013, 0.0107129271563633},
{0.43610751692580, 0.71185685817894, 0.165464120764703},
{3.48444943441647, 2.70987679499897, 0.00167681387824439},
{2.87324755559053, 3.37569611982342, 0.0152148024934994},
{2.75689897736462, 0.42925535850908, 0.00831865349421553},
{3.86387051378001, 3.588135962301, 0.000115003178162785},
{2.27270718949677, 2.65955756016997, 0.16152445175425},
{1.82296865461929, 0.40576954037085, 0.0909445510199988},
{0.95985691765319, 2.21196241672429, 0.77818905840432},
{2.11957382881538, 1.69084627995986, 0.153941202361293},
{0.73998874096199, 3.30367760930762, 0.506327506970041},
{2.32939345434527, 1.27488999559425, 0.063236729771379},
{2.56193637001839, 2.32299257789808, 0.0708109639852804},
{3.21325905768044, 1.01293748566251, 0.00218612391199958},
{1.64952193509078, 1.88460879113632, 0.415411998545276},
{2.52607064096398, 2.89322759057237, 0.0719043559006391},
{0.37519061487987, 3.88681386816008, 0.21669558160062},
{1.39608479319049, 1.55811486084902, 0.409556046291366},
{3.72050935565971, 0.77588503336903, 0.000146321560831018},
{1.67474663466559, 3.79237403545817, 0.220981694209407},
{3.14447897808731, 1.46703024212435, 0.00443929740357612},
{1.18879971108139, 1.77001877470641, 0.556868686267464},
{2.57117882330697, 0.96067298105002, 0.0237295024263727},
{1.02712829891293, 3.29841053693053, 0.543829340683327},
{2.194392233783, 1.81583137263447, 0.143955638114054},
{1.35281002175827, 2.630584295287, 0.725490961748661},
{3.84834996726093, 1.18772616793581, 0.000102526702993097},
{3.04852033507743, 1.62055604786625, 0.0076259033979074},
{2.835785739481, 1.16197468514135, 0.0114990951974411},
{1.59801729049022, 2.50821541715605, 0.582746807643732},
{2.66950024917477, 3.93593797924368, 0.0188819958546992},
{1.38274272677552, 1.49161224809746, 0.389622329123069},
{1.95979888396333, 1.98648928573506, 0.271939200551171},
{1.50506813754446, 2.76196908677508, 0.610773147540629},
{2.09725292858138, 3.47641085658139, 0.139317213288209},
{1.78361624730369, 0.70066162047457, 0.121946567054367},
{2.78785210707873, 0.55313095789242, 0.00819762338315817},
{3.92182669893881, 0.30662399891452, 3.26180265555479E-05},
{1.39459101471458, 3.840527740712, 0.285555538934434},
{0.74569429898301, 0.90845602201494, 0.251104305423187},
{3.033033373959, 0.86472849463222, 0.00413800535219111},
{3.95919472471792, 2.34246035582443, 0.000128494265114893},
{1.81708128953285, 3.12131641319052, 0.323402226225631},
{3.13513802307079, 0.76504385088687, 0.00248795466280997},
{3.01830466620585, 2.41319957319954, 0.0140921940378313},
{0.88175157908856, 2.51000546149677, 0.821693629630674},
{3.913232759366, 3.89512643243538, 6.55185673338936E-05},
{1.98908827425658, 2.853544843535, 0.283738138845764},
{1.55205953860471, 2.34654214420508, 0.602585543653066},
{2.13797142764972, 1.47395178619166, 0.12158499276512},
{2.14943498329945, 1.61461006813091, 0.134488966991641},
{3.61468763153876, 2.170680679467, 0.000820632175031198},
{3.00997992500429, 3.152315073495, 0.0108255742288722},
{2.42533550107045, 1.58141252717036, 0.0641588299604641},
{1.27958407273506, 2.687041837230, 0.748843102280906},
{0.75887306536521, 1.55414671286804, 0.450443261663259},
{2.70229026670621, 1.57296702463016, 0.0267772658325181},
{1.50203875433701, 1.338030800651, 0.304768171900962},
{0.85012438866545, 2.62087264999768, 0.805021817777604},
{3.18152721805534, 3.44726704753639, 0.00408792042702153},
{3.33915629732403, 0.70002352886368, 0.000946928014199353},
{3.44147693590295, 3.26015796665106, 0.00144997516779128},
{2.14601954457956, 3.28092139882057, 0.148584974910111},
{1.42820747639709, 0.66332469733043, 0.182024617059069},
{2.76250214496686, 1.830261345540, 0.0271518568421737},
{2.76687748643264, 3.44127290196281, 0.0211291461747445},
{0.73878544269567, 3.73529648336938, 0.342647604704539},
{0.90722495673858, 3.74561452838723, 0.360316085975723},
{2.26516580843953, 2.19148250319797, 0.155782175848903},
{1.66880573417731, 3.64232393625246, 0.255239869651753},
{2.60561677222271, 1.94575590902142, 0.0503755220620965},
{2.82605778908498, 2.82663953258345, 0.0272694879013201},
{2.49544805083318, 3.45748213442449, 0.0504760063520445},
{1.89908489594215, 1.11900162839307, 0.143407987697774},
{3.091990091533, 3.00717813358809, 0.00862612678751525},
{1.60735245209824, 2.421183133037, 0.573290933755564},
{3.89869697943765, 3.96166497638022, 6.72323590424165E-05},
{2.31411674442098, 2.40340056589418, 0.147051780669988},
{0.96015042852613, 1.68239314045813, 0.53434489526657},
{3.36424111343553, 3.41364894647003, 0.00183628427512259},
{3.09677169867716, 3.73976723387366, 0.00450160033889463},
{2.58033571161338, 1.19773077620141, 0.028417257018944},
{3.404974463443, 1.42140589335966, 0.00130182425485564},
{0.69508472694900, 3.65328026671272, 0.360157258757033},
{3.408407035146, 3.78032180661809, 0.00106587928881656},
{2.03701505191555, 0.90696195236103, 0.0912739866198576},
{0.46951130839751, 3.33566408495783, 0.397565668335093},
{3.06152477973642, 0.94973749652266, 0.00395910080965704},
{3.59192660627699, 3.00409599288928, 0.000831333544051649},
{3.73966880709391, 1.74661901862049, 0.000311099365247085},
{1.06916081108366, 3.790405692833, 0.347356786564297},
{0.30739091008608, 2.12786087538764, 0.462435765295237},
{3.12905329873996, 0.38518859755368, 0.00189509735975009},
{3.21223181700619, 3.42260815334101, 0.0036523027139464}
      };
      #endregion
      var n = 100;
      var y = new double[100];
      for (int i = 0; i < y.Length; i++) { y[i] = xy[i, 2]; }
      var y_mean = y.Average();
      var y_var = y.Select(yi => (yi - y_mean) * (yi - y_mean)).Sum() / (n - 1);

      int nBasisFunctionsPerVariable = 5;
      var nCols = nBasisFunctionsPerVariable * 2;


      void phiFunc(double[] alpha, ref double[,] phi) {
        // use multiple basis functions of the form exp(ax)
        if (phi == null) phi = new double[n, nCols + 1];
        int alphaIdx = 0;
        // basis functions for x1
        int i = 0;
        for (; i < 5; i++) {
          for (int r = 0; r < n; r++) {
            phi[r, i] = Math.Exp(alpha[alphaIdx] * xy[r, 0]);
          }
          alphaIdx++;
        }
        // basis functions for x2
        for (; i < 10; i++) {
          for (int r = 0; r < n; r++) {
            phi[r, i] = Math.Exp(alpha[alphaIdx] * xy[r, 1]);
          }
          alphaIdx++;
        }
        // constant
        for (int r = 0; r < n; r++) {
          phi[r, i] = 1.0;
        }
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        if (Jac == null) {
          Jac = new double[n, nCols];
          ind = new int[2, nCols];
          for (int j = 0; j < nCols; j++) {
            ind[0, j] = j; ind[1, j] = j;
          }
        }
        int alphaIdx = 0;
        // basis functions for x1
        int i = 0;
        for (; i < nCols / 2; i++) {
          for (int r = 0; r < n; r++) {
            Jac[r, i] = xy[r, 0] * Math.Exp(alpha[alphaIdx] * xy[r, 0]);
          }
          alphaIdx++;
        }
        // basis functions for x2
        for (; i < nCols; i++) {
          for (int r = 0; r < n; r++) {
            Jac[r, i] = xy[r, 1] * Math.Exp(alpha[alphaIdx] * xy[r, 1]);
          }
          alphaIdx++;
        }
      }
      var rand = new Random(1234);
      var alpha = Enumerable.Range(0, nCols).Select(_ => RandNormal(rand, 1.0)).ToArray(); // init alpha randomly;
      VarPro.VariableProjection.Fit(phiFunc, jacFunc, y, alpha, out var coeff, out var report, maxIters: 20, useWGCV: true);
      Console.WriteLine($"NMSE = {report.residNormSqr / n / y_var:e3}");

      var threshold = 0.1;
      Console.Write($"{coeff[0]:e4}");
      for (int i = 1; i <= 5; i++) {
        if (Math.Abs(coeff[i]) > threshold) Console.Write($" + {coeff[i]:e3} exp({alpha[i - 1]:e3} X1)");
      }
      for (int i = 6; i <= 10; i++) {
        if (Math.Abs(coeff[i]) > threshold) Console.Write($" + {coeff[i]:e3} exp({alpha[i - 1]:e3} X2)");
      }
    }

    // https://stackoverflow.com/questions/12983731/algorithm-for-calculating-binomial-coefficient
    private long Binom(uint n, uint k) {
      // This function gets the total number of unique combinations based upon N and K.
      // N is the total number of items.
      // K is the size of the group.
      // Total number of unique combinations = N! / ( K! (N - K)! ).
      // This function is less efficient, but is more likely to not overflow when N and K are large.
      // Taken from:  http://blog.plover.com/math/choose.html
      //
      long r = 1;
      long d;
      if (k > n) return 0;
      for (d = 1; d <= k; d++) {
        r *= n--;
        r /= d;
      }
      return r;
    }

    [TestMethod]
    public void UnivariateHingeFunctions() {
      // a test using parametric "hinge-function" f(x,a) = max(x - a, 0) and max(a - x, 0)

      // the target is sin(x) for x \in -pi .. pi
      var n = 20;
      var y = new double[n];
      var x = new double[n];
      for (int i = 0; i < n; i++) {
        x[i] = -Math.PI + i / (double)n * 2 * Math.PI;
        y[i] = Math.Sin(x[i]);
      }


      // the model is a sum of 6 hinge functions (3 positive, 3 negative) + a constant offset
      var nTerms = 6;
      void phiFunc(double[] alpha, ref double[,] phi) {
        if (phi == null) phi = new double[n, nTerms + 1];
        int alphaIdx = 0;
        // basis functions for x

        for (int i = 0; i < nTerms; i++) {
          for (int r = 0; r < n; r++) {
            if (i % 2 == 0)
              phi[r, i] = Math.Max(x[r] - alpha[alphaIdx], 0.0); // positive
            else
              phi[r, i] = Math.Max(alpha[alphaIdx] - x[r], 0.0); // negative
          }
          alphaIdx++;
        }
        // constant
        for (int r = 0; r < n; r++) {
          phi[r, nTerms] = 1.0;
        }
      }

      void jacFunc(double[] alpha, ref double[,] Jac, ref int[,] ind) {
        if (Jac == null) {
          Jac = new double[n, nTerms];
          ind = new int[2, nTerms];
          for (int j = 0; j < nTerms; j++) {
            ind[0, j] = j; ind[1, j] = j;
          }
        }
        int alphaIdx = 0;
        for (int i = 0; i < nTerms; i++) {
          for (int r = 0; r < n; r++) {
            if (i % 2 == 0)
              Jac[r, i] = x[r] > alpha[alphaIdx] ? 1.0 : 0.0;
            else
              Jac[r, i] = x[r] < alpha[alphaIdx] ? -1.0 : 0.0;
          }
          alphaIdx++;
        }
      }


      var rand = new Random(1234);
      var alpha = Enumerable.Range(0, nTerms).Select(_ => RandNormal(rand, 1.0)).ToArray(); // init alpha randomly;

      TestVarPro(alpha, y, phiFunc, jacFunc, phiFunc, y, out var yPred, useWGCV: false);
      Console.WriteLine("y[i]\tyPred[i]");
      for (int i = 0; i < y.Length; i++) {
        Console.WriteLine($"{y[i]}\t{yPred[i]}");
      }
    }


    private void TestVarPro(double[] alpha, double[] y, VariableProjection.FeatureFunc phiFunc, VariableProjection.JacobianFunc jacFunc,
    VariableProjection.FeatureFunc phiFuncTest, double[] yTest, out double[] yPred, bool useWGCV = true) {

      var sw = new Stopwatch();
      sw.Start();

      void writeProgress(VariableProjection.Report rep, CancellationToken cancelToken) {
        var coeffNormSqr = Dot(rep.coeff, rep.coeff);
        Console.WriteLine($"{rep.iter,3}{rep.residNormSqr + rep.lambda * coeffNormSqr,11:e3}{rep.residNormSqr,11:e3}{coeffNormSqr,11:e3}" +
                          $"{rep.lineSearchStep,10:e2}{rep.lambda,11:e3}{rep.w,11:e3}{rep.gradNorm,11:e3}{rep.lineSearchIterations,4} [{string.Join(" ", rep.alpha.Take(3).Select(ai => ai.ToString("e3")))}]");
      }

      Console.WriteLine($"{"It",3}{"     C     ",11}{"   ||r||²   ",11}{"  ||c||²  ",11}{"  step  ",10}{"    lam    ",11}{"     w     ",11}{"  gradNorm ",11}{"lsit",4}{"    alpha    ",21}");

      VariableProjection.Fit(phiFunc, jacFunc, y, alpha, out var coeff, out var report, iterationCallback: writeProgress, useWGCV: useWGCV);

      double[,] testPhi = null;
      phiFuncTest(alpha, ref testPhi);
      var m = yTest.Length;
      var n = testPhi.GetLength(1);
      var r = new double[m];
      yPred = new double[m];
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          yPred[i] += testPhi[i, j] * coeff[j];
        }
        r[i] = yTest[i] - yPred[i];
      }

      Console.WriteLine();

      Console.WriteLine($"||coeff||²       {Dot(coeff, coeff):e4}");
      Console.WriteLine($"||resid_train||² {report.residNormSqr:e4}");
      Console.WriteLine($"||resid_train||  {report.residNorm:e4}");
      Console.WriteLine($"MSE_train        {report.residNormSqr / r.Length:e4}");
      Console.WriteLine($"||resid_test||²  {Dot(r, r):e4}");
      Console.WriteLine($"||resid_test||   {Math.Sqrt(Dot(r, r)):e4}");
      Console.WriteLine($"MSE_test         {Dot(r, r) / r.Length:e4}");
      Console.WriteLine($"Runtime:         {sw.ElapsedMilliseconds}ms");


      // for (int i = 0; i < yTest.Length; i++) {
      //   Console.WriteLine($"{yTest[i]} {yPred[i]}");
      // }
    }

    private double RandNormal(Random random, double sigma) {
      // polar method
      double u, v, q, p;

      do {
        u = 2.0 * random.NextDouble() - 1;
        v = 2.0 * random.NextDouble() - 1;
        q = u * u + v * v;
      } while (q >= 1.0 || q == 0.0);

      p = Math.Sqrt(-2 * Math.Log(q) / q);
      return u * p * sigma;
      // x2 = v * p; ignored
    }

    private static double Dot(double[] a, double[] b) {
      if (a.Length != b.Length) throw new InvalidProgramException();
      var r = 0.0;
      for (int i = 0; i < a.Length; i++) r += a[i] * b[i];
      return r;
    }



    // https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/monthly-lake-erie-levels-1921-19.csv
    public static double[] MonthlyLakeErieLevels_1921_1970 = {
14.763,// "1921-01",
14.649,// "1921-02",
15.085,// "1921-03",
16.376,// "1921-04",
16.926,// "1921-05",
16.774,// "1921-06",
16.490,// "1921-07",
15.769,// "1921-08",
15.180,// "1921-09",
14.383,// "1921-10",
14.478,// "1921-11",
14.364,// "1921-12",
13.928,// "1922-01",
13.283,// "1922-02",
13.700,// "1922-03",
15.465,// "1922-04",
16.243,// "1922-05",
16.490,// "1922-06",
16.243,// "1922-07",
15.787,// "1922-08",
15.446,// "1922-09",
14.649,// "1922-10",
13.776,// "1922-11",
13.188,// "1922-12",
13.283,// "1923-01",
12.657,// "1923-02",
12.979,// "1923-03",
13.909,// "1923-04",
14.535,// "1923-05",
14.877,// "1923-06",
14.858,// "1923-07",
14.288,// "1923-08",
13.947,// "1923-09",
13.416,// "1923-10",
12.903,// "1923-11",
13.454,// "1923-12",
13.491,// "1924-01",
13.567,// "1924-02",
13.397,// "1924-03",
14.440,// "1924-04",
15.161,// "1924-05",
15.427,// "1924-06",
15.693,// "1924-07",
15.142,// "1924-08",
14.763,// "1924-09",
14.288,// "1924-10",
13.074,// "1924-11",
12.543,// "1924-12",
12.239,// "1925-01",
12.011,// "1925-02",
12.827,// "1925-03",
13.567,// "1925-04",
13.548,// "1925-05",
13.302,// "1925-06",
13.188,// "1925-07",
13.112,// "1925-08",
12.827,// "1925-09",
12.201,// "1925-10",
11.917,// "1925-11",
11.803,// "1925-12",
11.157,// "1926-01",
10.892,// "1926-02",
11.120,// "1926-03",
12.600,// "1926-04",
13.283,// "1926-05",
13.416,// "1926-06",
13.340,// "1926-07",
13.529,// "1926-08",
13.776,// "1926-09",
14.307,// "1926-10",
13.852,// "1926-11",
13.833,// "1926-12",
13.169,// "1927-01",
12.941,// "1927-02",
13.188,// "1927-03",
14.383,// "1927-04",
14.763,// "1927-05",
15.218,// "1927-06",
15.161,// "1927-07",
14.858,// "1927-08",
14.156,// "1927-09",
13.586,// "1927-10",
13.150,// "1927-11",
14.137,// "1927-12",
14.231,// "1928-01",
14.364,// "1928-02",
13.833,// "1928-03",
14.478,// "1928-04",
15.009,// "1928-05",
15.617,// "1928-06",
16.148,// "1928-07",
15.977,// "1928-08",
15.142,// "1928-09",
14.592,// "1928-10",
14.364,// "1928-11",
14.497,// "1928-12",
14.554,// "1929-01",
14.991,// "1929-02",
15.863,// "1929-03",
17.932,// "1929-04",
19.184,// "1929-05",
19.184,// "1929-06",
18.956,// "1929-07",
18.254,// "1929-08",
17.514,// "1929-09",
16.660,// "1929-10",
16.338,// "1929-11",
16.319,// "1929-12",
17.457,// "1930-01",
17.173,// "1930-02",
17.856,// "1930-03",
18.596,// "1930-04",
18.558,// "1930-05",
18.159,// "1930-06",
17.685,// "1930-07",
16.812,// "1930-08",
16.072,// "1930-09",
15.332,// "1930-10",
14.478,// "1930-11",
14.213,// "1930-12",
13.738,// "1931-01",
13.169,// "1931-02",
12.581,// "1931-03",
13.245,// "1931-04",
13.852,// "1931-05",
14.175,// "1931-06",
14.288,// "1931-07",
13.985,// "1931-08",
13.435,// "1931-09",
12.884,// "1931-10",
12.429,// "1931-11",
12.410,// "1931-12",
13.397,// "1932-01",
13.909,// "1932-02",
13.833,// "1932-03",
14.099,// "1932-04",
14.687,// "1932-05",
14.611,// "1932-06",
14.383,// "1932-07",
13.909,// "1932-08",
13.359,// "1932-09",
12.296,// "1932-10",
12.106,// "1932-11",
11.803,// "1932-12",
12.353,// "1933-01",
12.220,// "1933-02",
12.827,// "1933-03",
14.250,// "1933-04",
15.085,// "1933-05",
14.953,// "1933-06",
14.440,// "1933-07",
13.890,// "1933-08",
13.036,// "1933-09",
12.201,// "1933-10",
11.404,// "1933-11",
11.309,// "1933-12",
10.987,// "1934-01",
10.361,// "1934-02",
10.304,// "1934-03",
11.347,// "1934-04",
11.784,// "1934-05",
11.841,// "1934-06",
11.841,// "1934-07",
11.651,// "1934-08",
11.404,// "1934-09",
10.873,// "1934-10",
10.209,// "1934-11",
10.076,// "1934-12",
10.247,// "1935-01",
10.133,// "1935-02",
10.740,// "1935-03",
11.556,// "1935-04",
12.201,// "1935-05",
12.505,// "1935-06",
12.732,// "1935-07",
12.201,// "1935-08",
12.068,// "1935-09",
11.290,// "1935-10",
11.139,// "1935-11",
11.101,// "1935-12",
10.342,// "1936-01",
10.000,// "1936-02",
11.347,// "1936-03",
12.770,// "1936-04",
13.321,// "1936-05",
13.340,// "1936-06",
13.188,// "1936-07",
12.676,// "1936-08",
12.315,// "1936-09",
12.049,// "1936-10",
11.594,// "1936-11",
11.252,// "1936-12",
12.467,// "1937-01",
13.491,// "1937-02",
13.491,// "1937-03",
14.156,// "1937-04",
15.256,// "1937-05",
15.598,// "1937-06",
16.034,// "1937-07",
15.598,// "1937-08",
14.516,// "1937-09",
13.435,// "1937-10",
12.638,// "1937-11",
12.106,// "1937-12",
12.144,// "1938-01",
12.979,// "1938-02",
11.917,// "1938-03",
15.066,// "1938-04",
15.199,// "1938-05",
15.427,// "1938-06",
15.427,// "1938-07",
15.408,// "1938-08",
14.706,// "1938-09",
14.137,// "1938-10",
13.302,// "1938-11",
12.979,// "1938-12",
13.036,// "1939-01",
12.903,// "1939-02",
13.719,// "1939-03",
14.801,// "1939-04",
15.541,// "1939-05",
15.617,// "1939-06",
15.503,// "1939-07",
15.218,// "1939-08",
14.592,// "1939-09",
13.852,// "1939-10",
13.416,// "1939-11",
13.055,// "1939-12",
12.315,// "1940-01",
12.239,// "1940-02",
12.562,// "1940-03",
13.890,// "1940-04",
14.687,// "1940-05",
15.313,// "1940-06",
15.408,// "1940-07",
15.028,// "1940-08",
14.782,// "1940-09",
14.213,// "1940-10",
13.435,// "1940-11",
13.662,// "1940-12",
14.288,// "1941-01",
13.491,// "1941-02",
13.150,// "1941-03",
13.586,// "1941-04",
13.871,// "1941-05",
14.175,// "1941-06",
14.099,// "1941-07",
13.719,// "1941-08",
13.093,// "1941-09",
12.562,// "1941-10",
12.030,// "1941-11",
12.144,// "1941-12",
11.860,// "1942-01",
12.410,// "1942-02",
12.770,// "1942-03",
14.630,// "1942-04",
15.218,// "1942-05",
15.920,// "1942-06",
15.882,// "1942-07",
15.750,// "1942-08",
15.275,// "1942-09",
14.801,// "1942-10",
14.554,// "1942-11",
14.345,// "1942-12",
15.104,// "1943-01",
14.554,// "1943-02",
14.953,// "1943-03",
15.958,// "1943-04",
17.590,// "1943-05",
18.805,// "1943-06",
18.805,// "1943-07",
18.330,// "1943-08",
17.476,// "1943-09",
16.812,// "1943-10",
16.224,// "1943-11",
15.541,// "1943-12",
14.744,// "1944-01",
14.478,// "1944-02",
14.725,// "1944-03",
16.319,// "1944-04",
17.324,// "1944-05",
17.609,// "1944-06",
17.211,// "1944-07",
16.546,// "1944-08",
15.977,// "1944-09",
15.427,// "1944-10",
14.972,// "1944-11",
14.535,// "1944-12",
14.213,// "1945-01",
13.719,// "1945-02",
15.009,// "1945-03",
16.319,// "1945-04",
17.078,// "1945-05",
17.913,// "1945-06",
18.159,// "1945-07",
17.704,// "1945-08",
17.059,// "1945-09",
17.268,// "1945-10",
16.546,// "1945-11",
16.072,// "1945-12",
15.996,// "1946-01",
15.294,// "1946-02",
15.901,// "1946-03",
16.300,// "1946-04",
16.546,// "1946-05",
17.495,// "1946-06",
17.666,// "1946-07",
16.964,// "1946-08",
16.148,// "1946-09",
15.427,// "1946-10",
14.839,// "1946-11",
14.269,// "1946-12",
14.118,// "1947-01",
14.156,// "1947-02",
14.080,// "1947-03",
16.414,// "1947-04",
18.216,// "1947-05",
19.298,// "1947-06",
18.824,// "1947-07",
18.368,// "1947-08",
17.875,// "1947-09",
16.869,// "1947-10",
16.129,// "1947-11",
15.712,// "1947-12",
15.617,// "1948-01",
15.161,// "1948-02",
16.148,// "1948-03",
17.609,// "1948-04",
18.406,// "1948-05",
18.463,// "1948-06",
18.254,// "1948-07",
17.609,// "1948-08",
16.812,// "1948-09",
15.712,// "1948-10",
15.066,// "1948-11",
14.763,// "1948-12",
14.877,// "1949-01",
15.370,// "1949-02",
15.731,// "1949-03",
15.996,// "1949-04",
16.224,// "1949-05",
16.186,// "1949-06",
16.015,// "1949-07",
15.446,// "1949-08",
14.573,// "1949-09",
14.080,// "1949-10",
13.226,// "1949-11",
12.979,// "1949-12",
14.459,// "1950-01",
15.655,// "1950-02",
15.636,// "1950-03",
17.154,// "1950-04",
17.381,// "1950-05",
17.116,// "1950-06",
16.736,// "1950-07",
16.167,// "1950-08",
15.920,// "1950-09",
15.408,// "1950-10",
15.066,// "1950-11",
15.769,// "1950-12",
15.787,// "1951-01",
15.863,// "1951-02",
17.154,// "1951-03",
18.178,// "1951-04",
18.653,// "1951-05",
18.615,// "1951-06",
18.406,// "1951-07",
17.837,// "1951-08",
17.078,// "1951-09",
16.471,// "1951-10",
16.224,// "1951-11",
16.395,// "1951-12",
17.400,// "1952-01",
18.672,// "1952-02",
19.089,// "1952-03",
19.829,// "1952-04",
20.000,// "1952-05",
19.943,// "1952-06",
19.526,// "1952-07",
18.975,// "1952-08",
18.463,// "1952-09",
17.268,// "1952-10",
16.414,// "1952-11",
16.414,// "1952-12",
16.755,// "1953-01",
16.736,// "1953-02",
17.173,// "1953-03",
17.647,// "1953-04",
18.216,// "1953-05",
18.767,// "1953-06",
18.539,// "1953-07",
18.273,// "1953-08",
17.419,// "1953-09",
16.679,// "1953-10",
15.996,// "1953-11",
15.465,// "1953-12",
15.408,// "1954-01",
15.123,// "1954-02",
16.072,// "1954-03",
17.685,// "1954-04",
18.235,// "1954-05",
18.064,// "1954-06",
17.818,// "1954-07",
17.438,// "1954-08",
16.812,// "1954-09",
17.116,// "1954-10",
17.211,// "1954-11",
17.097,// "1954-12",
17.495,// "1955-01",
17.097,// "1955-02",
18.121,// "1955-03",
18.748,// "1955-04",
18.767,// "1955-05",
18.539,// "1955-06",
18.102,// "1955-07",
17.818,// "1955-08",
17.002,// "1955-09",
16.357,// "1955-10",
15.560,// "1955-11",
15.313,// "1955-12",
14.782,// "1956-01",
13.719,// "1956-02",
14.839,// "1956-03",
15.825,// "1956-04",
17.324,// "1956-05",
17.723,// "1956-06",
17.685,// "1956-07",
17.514,// "1956-08",
16.907,// "1956-09",
15.863,// "1956-10",
14.934,// "1956-11",
14.706,// "1956-12",
14.383,// "1957-01",
14.345,// "1957-02",
14.763,// "1957-03",
15.958,// "1957-04",
16.698,// "1957-05",
16.793,// "1957-06",
17.230,// "1957-07",
16.509,// "1957-08",
15.787,// "1957-09",
14.991,// "1957-10",
14.080,// "1957-11",
14.326,// "1957-12",
14.497,// "1958-01",
13.510,// "1958-02",
13.662,// "1958-03",
14.042,// "1958-04",
14.269,// "1958-05",
14.478,// "1958-06",
14.972,// "1958-07",
14.972,// "1958-08",
14.573,// "1958-09",
13.776,// "1958-10",
13.017,// "1958-11",
12.600,// "1958-12",
12.296,// "1959-01",
13.131,// "1959-02",
13.966,// "1959-03",
15.028,// "1959-04",
15.844,// "1959-05",
15.769,// "1959-06",
15.237,// "1959-07",
14.801,// "1959-08",
14.137,// "1959-09",
13.890,// "1959-10",
13.416,// "1959-11",
13.871,// "1959-12",
14.478,// "1960-01",
14.725,// "1960-02",
14.763,// "1960-03",
15.806,// "1960-04",
16.565,// "1960-05",
17.097,// "1960-06",
17.306,// "1960-07",
17.211,// "1960-08",
16.717,// "1960-09",
15.787,// "1960-10",
14.801,// "1960-11",
14.231,// "1960-12",
13.966,// "1961-01",
14.004,// "1961-02",
15.313,// "1961-03",
16.357,// "1961-04",
17.742,// "1961-05",
17.609,// "1961-06",
17.249,// "1961-07",
17.078,// "1961-08",
16.509,// "1961-09",
15.465,// "1961-10",
14.706,// "1961-11",
14.213,// "1961-12",
13.662,// "1962-01",
13.928,// "1962-02",
14.516,// "1962-03",
15.180,// "1962-04",
15.351,// "1962-05",
15.579,// "1962-06",
15.446,// "1962-07",
15.199,// "1962-08",
14.725,// "1962-09",
14.383,// "1962-10",
14.231,// "1962-11",
13.776,// "1962-12",
13.150,// "1963-01",
12.713,// "1963-02",
13.283,// "1963-03",
14.478,// "1963-04",
14.858,// "1963-05",
14.782,// "1963-06",
14.307,// "1963-07",
14.023,// "1963-08",
13.529,// "1963-09",
12.922,// "1963-10",
12.410,// "1963-11",
11.936,// "1963-12",
11.784,// "1964-01",
11.992,// "1964-02",
12.619,// "1964-03",
13.662,// "1964-04",
14.231,// "1964-05",
14.099,// "1964-06",
13.833,// "1964-07",
13.416,// "1964-08",
12.922,// "1964-09",
11.974,// "1964-10",
11.461,// "1964-11",
11.423,// "1964-12",
11.860,// "1965-01",
12.163,// "1965-02",
13.226,// "1965-03",
13.966,// "1965-04",
14.535,// "1965-05",
14.516,// "1965-06",
14.231,// "1965-07",
13.966,// "1965-08",
13.738,// "1965-09",
13.226,// "1965-10",
12.998,// "1965-11",
13.131,// "1965-12",
13.700,// "1966-01",
13.814,// "1966-02",
14.383,// "1966-03",
15.047,// "1966-04",
15.693,// "1966-05",
15.844,// "1966-06",
15.712,// "1966-07",
15.313,// "1966-08",
14.763,// "1966-09",
13.548,// "1966-10",
13.586,// "1966-11",
14.516,// "1966-12",
14.383,// "1967-01",
14.497,// "1967-02",
14.744,// "1967-03",
15.806,// "1967-04",
16.527,// "1967-05",
16.546,// "1967-06",
16.717,// "1967-07",
16.433,// "1967-08",
15.769,// "1967-09",
15.275,// "1967-10",
15.123,// "1967-11",
15.541,// "1967-12",
15.825,// "1968-01",
16.376,// "1968-02",
16.357,// "1968-03",
16.907,// "1968-04",
17.021,// "1968-05",
17.419,// "1968-06",
17.533,// "1968-07",
17.268,// "1968-08",
16.603,// "1968-09",
15.825,// "1968-10",
15.446,// "1968-11",
15.636,// "1968-12",
15.693,// "1969-01",
16.755,// "1969-02",
16.509,// "1969-03",
17.723,// "1969-04",
18.691,// "1969-05",
19.127,// "1969-06",
19.564,// "1969-07",
19.203,// "1969-08",
18.216,// "1969-09",
17.211,// "1969-10",
16.660,// "1969-11",
16.831,// "1969-12",
15.769,// "1970-01",
15.731,// "1970-02",
15.996,// "1970-03",
17.021,// "1970-04",
17.552,// "1970-05",
17.837,// "1970-06",
17.856,// "1970-07",
17.571,// "1970-08",
17.078,// "1970-09",
16.660,// "1970-10",
16.433,// "1970-11",
16.584// "1970-12",
    };
  }
}
