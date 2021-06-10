using System;
using System.Linq;
using System.Threading;

namespace HEAL.VarPro.Demo {
  class Program {
    static void Main(string[] args) {
      TestExponentialModel();
    }

    private static void TestExponentialModel() {
      // Example from
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

      var random = new Random(1234);
      for (int i = 0; i < N; i++) {
        yTest[i] = c[0] * phi[i, 0]
               + c[1] * phi[i, 1];
        y[i] = yTest[i] + RandNormal(random, noiseSigma);
      }

      void writeProgress(VariableProjection.Report rep, CancellationToken cancelToken) {
        var coeffNormSqr = Dot(rep.coeff, rep.coeff);
        Console.WriteLine($"{rep.iter} ||r||² {rep.residNormSqr:e3} ||coeff||² {coeffNormSqr:e3} obj. {rep.residNormSqr + rep.lambda * coeffNormSqr:e3} line-search step: {rep.lineSearchStep:e2} lam: {rep.lambda:e3} w: {rep.w:e3} a: [{string.Join(" ", rep.alpha.Select(ai => ai.ToString("e3")))}]");
      }

      double[] initialAlpha = new[] { 0.8457, 2.3331, 7.4757 };
      var alpha = (double[])initialAlpha.Clone();
      VariableProjection.Fit(phiFunc, jacFunc, y, alpha, out var coeff, iterationCallback: writeProgress);

      var yPred = new double[N];
      var r = new double[N];
      var n = phi.GetLength(1);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < n; j++) {
          yPred[i] += phi[i, j] * coeff[j];
        }
        r[i] = yTest[i] - yPred[i];
      }

      Console.WriteLine($"Optimal    alpha: {string.Join(" ", optAlpha.Select(ai => ai.ToString("e3")))}");
      Console.WriteLine($"Identified alpha: {string.Join(" ", alpha.Select(ai => ai.ToString("e3")))}");
      Console.WriteLine();
      Console.WriteLine($"||coeff||²        {Dot(coeff, coeff)}");
      Console.WriteLine($"||resid_test||²   {Dot(r, r)}");
      Console.WriteLine($"||resid_test||    {Math.Sqrt(Dot(r, r))}");
      Console.WriteLine($"MSE (test)        {Dot(r, r) / r.Length}");


      // compare to result from paper
      Console.WriteLine();
      Console.WriteLine("Reported by Chen et al.: w: 0.95, lambda: 0.098");
      alpha = new[] { 1.06, 1.01, 14.98 };
      phiFunc(alpha, ref phi);
      coeff = new[] { 2.99, 5.06 };
      Array.Clear(yPred, 0, yPred.Length);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < n; j++) {
          yPred[i] += phi[i, j] * coeff[j];
        }
        r[i] = yTest[i] - yPred[i];
      }
      Console.WriteLine($"||coeff||²        {Dot(coeff, coeff)}");
      Console.WriteLine($"||resid_test||²   {Dot(r, r)}");
      Console.WriteLine($"||resid_test||    {Math.Sqrt(Dot(r, r))}");
      Console.WriteLine($"MSE (test)        {Dot(r, r) / r.Length}");

      // for (int i = 0; i < yTest.Length; i++) {
      //   Console.WriteLine($"{yTest[i]} {yPred[i]}");
      // }
      //
    }

    private static double RandNormal(Random random, double noiseSigma) {
      // polar method
      double u, v, q, p;

      do {
        u = 2.0 * random.NextDouble() - 1;
        v = 2.0 * random.NextDouble() - 1;
        q = u * u + v * v;
      } while (q >= 1.0 || q == 0.0);

      p = Math.Sqrt(-2 * Math.Log(q) / q);
      return u * p * noiseSigma;
      // x2 = v * p; ignored
    }



    private static double Dot(double[] a, double[] b) {
      if (a.Length != b.Length) throw new InvalidProgramException();
      var r = 0.0;
      for (int i = 0; i < a.Length; i++) r += a[i] * b[i];
      return r;
    }
  }
}
