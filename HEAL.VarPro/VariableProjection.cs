using System;
using System.Linq;
using System.Threading;

namespace HEAL.VarPro {
  public class VariableProjection {
    public delegate void FeatureFunc(double[] alpha, ref double[,] phi);
    public delegate void JacobianFunc(double[] alpha, ref double[,] Jac, ref int[,] ind);
    public class Report {
      public int iter;
      public double residNorm;
      public double residNormSqr;
      public double lambda;
      public double w;
      public double lineSearchStep;
      public double lineSearchIterations;
      public double[] alpha;
      public double[] coeff;
      public double gradNorm;
    }

    private VariableProjection(FeatureFunc Phi, JacobianFunc Jac, int nAlpha, double[] y, Action<Report, CancellationToken> iterationCallback = null) {
      this.y = y;
      this.Phi = Phi;
      this.Jac = Jac;
      this.iterationCallback = iterationCallback;
      this.r = new double[y.Length]; // residuals
      this.m = y.Length;
      this.nAlpha = nAlpha;
      J = new double[m, nAlpha];
      Jac2 = new double[m, nAlpha];
      grad = new double[nAlpha];
      U = new double[m, m];
    }

    public bool WGCV { get; set; } = true;

    private readonly double[] y;
    private readonly FeatureFunc Phi;
    private readonly JacobianFunc Jac;


    private Action<Report, CancellationToken> iterationCallback;
    private double eps;
    private int maxIters;
    private int nAlpha;
    private int m;

    // buffers
    private double[] r, s, alpha, nextAlpha, coeff, grad;
    private double[,] F, dF, J, U, VT, Jac2;
    private int[,] ind;

    public static void Fit(FeatureFunc Phi, JacobianFunc Jac, double[] y, double[] alpha, out double[] coeff, out Report report,
      int maxIters = 20, Action<Report, CancellationToken> iterationCallback = null, bool useWGCV = true, double eps = 1e-16) {
      // Step 1: Choose theta_N_0 \in R^k, a small positive number eps and 
      //         the maximum iteration N. Set k = 0;

      var varpro = new VariableProjection(Phi, Jac, alpha.Length, y);
      varpro.WGCV = useWGCV;
      varpro.iterationCallback = iterationCallback;
      varpro.eps = eps;
      varpro.maxIters = maxIters;
      varpro.alpha = alpha; // don't copy alpha. Fit() updates alpha

      varpro.FitInternal(out report);
      coeff = varpro.coeff;
    }
    private void FitInternal(out Report report) {
      var w = 1.0; var lambda = 0.0;
      double fx, nextFx;
      double prevResNormSqr = double.PositiveInfinity, prevCoeffNormSqr = double.PositiveInfinity; // on iteration k=0 there is no previous result
      double resNormSqr, coeffNormSqr;
      double gradNorm = 0.0;
      nextAlpha = new double[alpha.Length];

      var cancellationTokenSource = new CancellationTokenSource();
      report = null;
      int k = 0;
      while (k <= maxIters) {
        var linesearch = false;
        double[] d = null; // line search direction

        Array.Copy(alpha, nextAlpha, nextAlpha.Length);

        // Backtracking line search (Boyd, Convex optimization, Chapter 9)
        // given descent direction d, for f a x \in dom(f), a \in (0, 0.5), b \in (0, 1)

        // line search parameters a and b
        // recommended settings from Boyd, Chapter 9
        var a = 0.1;
        var b = 0.5;
        var predictedChange = 0.0;
        var t = 1.0;
        var lineSearchEps = 1e-12; // smallest step for line search
        int lineSearchIterations = 0;

        // line search loop
        do {
          // Preparatory step: SVD
          // For step 2 and step 3 we use the SVD of Phi
          Phi(nextAlpha, ref F);
          var n = F.GetLength(1);
          var s_full = new double[n];
          alglib.svd.rmatrixsvd(F, m, n, uneeded: 2, vtneeded: 2, additionalmemory: 2, w: ref s, u: ref U, vt: ref VT, null);
          var tol = m * 2.2204460492503131E-16;  // the difference between 1.0 and the next larger double value
          var s0 = s[0];
          var rank = s.Count(si => si > tol * s0); // use rank cut-off

          // left-apply U to y
          var uy = new double[m];
          alglib.rmatrixgemv(m, m, alpha: 1.0, a: U, ia: 0, ja: 0, opa: 1,
            x: y, ix: 0,
            beta: 0.0, y: ref uy, iy: 0);


          // Step 2: Calculate the weight parameter w_k and
          //         regularization parameter lambda_k using (24) and (23)
          if (WGCV) {
            w = (k == 0) ? 1.0 : CalculateW(lambda, s, rank, uy);
            lambda = CalculateLambda(w, lambda, s, rank, uy);
          }

          // Step 3: Compute the linear parameters coeff using (16).
          //         Obtain the residual vector r and its approximated
          //         Jacobian matrix J using J_GP, J_Kau, or J_R.

          // eqn 16. 
          // (using SVD)
          // Golub & Van Loan, Matrix Computations, 4th Edition, Section 6.1.4 Ridge Regression, page 307.
          if (coeff == null) coeff = new double[n];
          double[] temp = new double[rank];
          for (int i = 0; i < rank; i++) temp[i] = s[i] * uy[i] / (s[i] * s[i] + lambda);
          alglib.rmatrixgemv(n, rank, 1.0, VT, 0, 0, 1, temp, 0, 0.0, ref coeff, 0);

          // calculate residuals
          r = (double[])y.Clone();
          alglib.rmatrixgemv(m, n, -1.0, F, 0, 0, 0, coeff, 0, 1.0, ref r, 0); // r = y - F coeff

          resNormSqr = 0.0;
          for (int i = 0; i < r.Length; i++) resNormSqr += r[i] * r[i];

          coeffNormSqr = 0.0;
          for (int i = 0; i < coeff.Length; i++) coeffNormSqr += coeff[i] * coeff[i];

          if (!linesearch) {
            lineSearchIterations--; // do not count this iteration as line-search iteration
            CalculateJacobian(Jac, nextAlpha, coeff, U, s, VT, rank, r, ref J);

            // Step 4: Calculate the gradient of the objective function (17)
            //         with respect to the nonlinear parameters
            //         ∇C = - J^T r
            alglib.rmatrixgemv(nAlpha, m, -1.0, J, 0, 0, 1, r, 0, 0.0, ref grad, 0);
            // Step 5: if ||∇C||^2 < eps, terminate the algorithm
            gradNorm = 0.0;
            for (int i = 0; i < grad.Length; i++) gradNorm += grad[i] * grad[i];


            // Step 6: Calculate the search direction d using (20), then
            //         update the nonlinear parameters theta_N by the line
            //         search procedure and (21)
            d = SolveLS(J, r);
            for (int i = 0; i < grad.Length; i++) predictedChange += grad[i] * d[i];
            if (predictedChange >= 0) throw new InvalidProgramException("Descent direction does not improve objective. Check your Jacobian calculation");
          }


          for (int i = 0; i < nextAlpha.Length; i++) nextAlpha[i] = alpha[i] - t * d[i];

          fx = prevResNormSqr + lambda * prevCoeffNormSqr; // C(lambda_k, theta_k-1)
          nextFx = resNormSqr + lambda * coeffNormSqr; // C(lambda_k, theta_k)

          t = b * t; // prepare for next line search iteration
          lineSearchIterations++;
        } while (t > lineSearchEps && nextFx > fx + a * t / b * predictedChange);   // end of line search loop

        if (gradNorm < eps || t <= lineSearchEps || cancellationTokenSource.IsCancellationRequested) break;

        Array.Copy(nextAlpha, alpha, alpha.Length); // apply the step
        prevResNormSqr = resNormSqr;
        prevCoeffNormSqr = resNormSqr;

        // Step 7: If k > N, terminate the algorithm; else k = k + 1,
        //         turn to step 2.
        k++;


        report = new Report() {
          iter = k,
          lineSearchIterations = lineSearchIterations,
          lineSearchStep = t / b,
          residNorm = Math.Sqrt(prevResNormSqr),
          lambda = lambda,
          residNormSqr = prevResNormSqr,
          w = w,
          alpha = (double[])alpha.Clone(),
          coeff = (double[])coeff.Clone(),
          gradNorm = gradNorm
        };

        iterationCallback?.Invoke(report, cancellationTokenSource.Token);
      }
    }



    // set full=false to drop the second term from the Jacobian (Kaufmann)
    private void CalculateJacobian(JacobianFunc Jac, double[] alpha, double[] coeff, double[,] U, double[] s, double[,] VT, int rank, double[] resid, ref double[,] J, bool full = true) {
      // code is based on VarPro.m by O'Leary and Rust
      // https://www.cs.umd.edu/~oleary/software/varpro/

      int numL = coeff.Length;
      int numN = alpha.Length;

      Jac(alpha, ref dF, ref ind);
      var p = dF.GetLength(1);
      double[] dF_r = new double[p]; // TODO allocate once
      alglib.rmatrixgemv(p, m, 1.0, dF, 0, 0, 1, resid, 0, 0.0, ref dF_r, 0);
      var T2 = new double[numL, numN]; // TODO allocate once

      Array.Clear(J, 0, J.Length);
      for (int j = 0; j < numN; j++) {                        // for each nonlinear parameter 
        for (int col = 0; col < ind.GetLength(1); col++) {
          if (ind[1, col] == j) {                             // columns of dPhi relevant to alpha(j)
            var termIdx = ind[0, col];                        // relevant element of coeff
            for (int k = 0; k < m; k++) {
              J[k, j] += coeff[termIdx] * dF[k, col];
            }
            T2[termIdx, j] = dF_r[col];
          }
        }
      }


      double[,] tmp = new double[m - rank, numN];
      alglib.rmatrixgemm(m - rank, numN, m, 1.0, U, 0, rank, 1, J, 0, 0, 0, 0.0, ref tmp, 0, 0);
      alglib.rmatrixgemm(m, numN, m - rank, -1.0, U, 0, rank, 0, tmp, 0, 0, 0, 0.0, ref J, 0, 0);


      double[,] tempT2 = new double[rank, numN];
      alglib.rmatrixgemm(rank, numN, numL, 1.0, VT, 0, 0, 0, T2, 0, 0, 0, 0.0, ref tempT2, 0, 0);

      for (int i = 0; i < rank; i++)
        for (int j = 0; j < numN; j++)
          tempT2[i, j] *= 1.0 / s[i];

      alglib.rmatrixgemm(m, numN, rank, -1.0, U, 0, 0, 0, tempT2, 0, 0, 0, 0.0, ref Jac2, 0, 0);

      if (full)
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < numN; j++) {
            J[i, j] += Jac2[i, j];
          }
        }
    }

    private static double[] SolveLS(double[,] A, double[] y) {
      var m = A.GetLength(0);
      var n = A.GetLength(1);
      alglib.rmatrixsolvels(A, m, n, y, 0.0, out var info, out var _, out var x);
      if (info < 0) throw new InvalidProgramException("Problem when solving linear system (LS)");
      return x;
    }

    // Since we do not know lambda_opt we use the regularization parameter lambda_k obtained
    // in the last iteration instead of lambda_opt. The initial value w_0 = 1.
    // Solve Eqn (24). See Appendix
    // length of s is limited to the rank
    private static double CalculateW(double lambda_opt, double[] s, int rank, double[] uy) {
      var lO = lambda_opt;
      var m = uy.Length;
      double h1(double lambda) {
        var t1 = 0.0;
        for (int i = 0; i < rank; i++) {
          t1 += lambda * lambda * uy[i] * uy[i] / (s[i] * s[i] + lambda);
        }
        var t2 = 0.0;
        for (int i = rank; i < m; i++) {
          t2 += uy[i] * uy[i];
        }
        return t1 + t2;
      }

      // Eq 36
      var s1 = 0.0; // numerator sum
      var s2 = 0.0; // first term in denominator
      var s3 = 0.0; // second term in denominator

      var f3 = 0.0;
      for (int i = 0; i < rank; i++) {
        var div = s[i] * s[i] + lO;
        s2 += s[i] * s[i] / (div * div);
        f3 += s[i] * s[i] / div; // Tikhonov filter factor 
      }

      for (int i = 0; i < rank; i++) {
        var suy = s[i] * uy[i];
        var div = s[i] * s[i] + lO;
        s1 += lO * suy * suy / (div * div * div);
        s3 += s[i] * s[i] * lO * uy[i] * uy[i] / (div * div * div);
      }

      return m * s1 / (s2 * h1(lO) + f3 * s3);
    }

    // length of s is limited to rank
    private static double CalculateLambda(double w, double lambda, double[] s, int rank, double[] uy) {
      var m = uy.Length;
      // eqn (23)
      void G(double[] x, ref double func, object _) {
        var lambda = x[0];
        var t1 = 0.0; // first term in numerator
        var t2 = 0.0;
        var dt1 = 0.0; // first term in denominator
        var dt2 = 0.0;
        double div;
        for (int i = 0; i < rank; i++) {
          div = 1.0 / (s[i] * s[i] + lambda);
          var t1_i = lambda * uy[i] * div;
          t1 += t1_i * t1_i;

          dt1 += (1 - w) * s[i] * s[i] * div; // (1-w) varphi_i 
          dt2 += lambda * div;
        }
        for (int i = rank; i < m; i++) {
          t2 += uy[i] * uy[i];
        }

        div = 1.0 / (dt1 + dt2 + m - rank);
        func = rank * (t1 + t2) * div * div;
      }

      // When w is fixed the optimal lambda can be obtained by solving (23) easily
      // using standard minimization algorithms.
      double[] x = new double[] { lambda }; // initial value
      alglib.minbccreatef(x, 1e-6, out var state);
      alglib.minbcsetbc(state, new double[] { 0 }, new double[] { double.PositiveInfinity });
      alglib.minbcoptimize(state, G, null, null);
      alglib.minbcresults(state, out var lambda_opt, out var report);
      if (report.terminationtype < 0) throw new InvalidProgramException();
      return lambda_opt[0];
    }
  }
}
