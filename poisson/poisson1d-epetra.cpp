// @HEADER
//
//    Epetra test for the Poisson problem.
//    Copyright (C) 2012--2014  Nico Schl\"omer
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// @HEADER
#include <string>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

// Epetra includes.
#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

#include <ml_epetra_preconditioner.h>

//#include "BelosConfigDefs.hpp"
#include <BelosLinearProblem.hpp>
#include <BelosEpetraAdapter.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosMinresSolMgr.hpp>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

using Teuchos::rcp;
using Teuchos::RCP;

// =============================================================================
typedef double                           ST;
typedef Epetra_MultiVector               MV;
typedef Epetra_Operator                  OP;
typedef Belos::MultiVecTraits<ST, MV>     MVT;
typedef Belos::OperatorTraits<ST, MV, OP>  OPT;

enum Operator {JAC, KEO, KEOREG, POISSON1D};
// =============================================================================
RCP<Epetra_CrsMatrix>
contructEpetraMatrix(const int n, const Epetra_Comm & eComm);
// =============================================================================
int main (int argc, char *argv[])
{
  Teuchos::GlobalMPISession(&argc, &argv, NULL);

  // Create a communicator for Epetra objects
#ifdef HAVE_MPI
  Epetra_MpiComm eComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm eComm();
#endif

  const RCP<Teuchos::FancyOStream> out =
    Teuchos::VerboseObjectBase::getDefaultOStream();

  bool success = true;
  try {
    // ===========================================================================
    // handle command line arguments
    Teuchos::CommandLineProcessor My_CLP;

    My_CLP.setDocString("Linear solver testbed for the 1D Poisson matrix.\n");

    std::string action("matvec");
    My_CLP.setOption("action",
                     &action,
                     "Which action to perform with the operator (matvec, solve_cg, solve_minres, solve_gmres)"
                    );

    std::string solver("cg");
//       My_CLP.setOption("solver", &solver, "Krylov subspace method (cg, minres, gmres)");

//       Operator op = JAC;
//       Operator allOpts[] = {JAC, KEO, KEOREG, POISSON1D};
//       std::string allOptNames[] = {"jac", "keo", "keoreg", "poisson1d"};
//       My_CLP.setOption("operator", &op, 4, allOpts, allOptNames);

    bool verbose = true;
    My_CLP.setOption("verbose", "quiet",
                     &verbose,
                     "Print messages and results.");

    int frequency = 10;
    My_CLP.setOption("frequency",
                     &frequency,
                     "Solvers frequency for printing residuals (#iters).");

    // Make sure this value is large enough to keep the cores busy for a while.
    int n = 1000;
    My_CLP.setOption("size",
                     &n,
                     "Size of the equation system (default: 1000).");

    // print warning for unrecognized arguments
    My_CLP.recogniseAllOptions(true);
    My_CLP.throwExceptions(false);

    // finally, parse the command line
    TEUCHOS_ASSERT_EQUALITY(My_CLP.parse (argc, argv),
                            Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL
                           );
    // =========================================================================
    // Construct Epetra matrix.
    RCP<Teuchos::Time> matrixConstructTime =
      Teuchos::TimeMonitor::getNewTimer("Epetra matrix construction");
    RCP<Epetra_CrsMatrix> A;
    {
      Teuchos::TimeMonitor tm(*matrixConstructTime);
      A = contructEpetraMatrix(n, eComm);
    }
//       A->Print(std::cout);

    // create initial guess and right-hand side
    RCP<Epetra_Vector> x =
      rcp(new Epetra_Vector(A->OperatorDomainMap()));
    RCP<Epetra_MultiVector> b =
      rcp(new Epetra_Vector(A->OperatorRangeMap()));
    // b->Random();
    TEUCHOS_ASSERT_EQUALITY(0, b->PutScalar(1.0));

    if (action.compare("matvec") == 0) {
      TEUCHOS_ASSERT_EQUALITY(0, x->PutScalar(1.0));
      RCP<Teuchos::Time> mvTime = Teuchos::TimeMonitor::getNewTimer("Epetra operator apply");
      {
        Teuchos::TimeMonitor tm(*mvTime);
        // Don't TEUCHOS_ASSERT_EQUALITY() here for speed.
        A->Apply(*x, *b);
      }

      // print timing data
      Teuchos::TimeMonitor::summarize();
    } else {
      // -----------------------------------------------------------------------
      // Belos part
      Teuchos::ParameterList belosList;
      // Relative convergence tolerance requested
      belosList.set("Convergence Tolerance", 1.0e-12);
      if (verbose) {
        belosList.set("Verbosity",
                      Belos::Errors +
                      Belos::Warnings +
                      Belos::IterationDetails +
                      Belos::FinalSummary +
                      Belos::Debug +
                      Belos::TimingDetails +
                      Belos::StatusTestDetails
                     );
        if (frequency > 0)
          belosList.set("Output Frequency", frequency);
      } else {
        belosList.set("Verbosity", Belos::Errors + Belos::Warnings);
      }

      // Belos::General, Belos::Brief
      belosList.set("Output Style", static_cast<int>(Belos::Brief));
      belosList.set("Maximum Iterations", 1000);

      // Construct an unpreconditioned linear problem instance.
      Belos::LinearProblem<double, MV, OP> problem(A, x, b);
      bool set = problem.setProblem();
      TEUCHOS_TEST_FOR_EXCEPTION(
          !set,
          std::logic_error,
          "ERROR:  Belos::LinearProblem failed to set up correctly!"
          );
      // -----------------------------------------------------------------------
      // Create an iterative solver manager.
      RCP<Belos::SolverManager<double, MV, OP> > newSolver;
      if (action.compare("solve_cg") == 0) {
        belosList.set("Assert Positive Definiteness", false);
        newSolver =
          rcp(new Belos::PseudoBlockCGSolMgr<double, MV, OP>(
                rcp(&problem, false),
                rcp(&belosList, false)
                ));
      } else if (action.compare("solve_minres") == 0) {
        newSolver =
          rcp(new Belos::MinresSolMgr<double, MV, OP>(
                rcp(&problem, false),
                rcp(&belosList, false)
                ));
      } else if (action.compare("solve_gmres") == 0) {
        newSolver =
          rcp(new Belos::PseudoBlockGmresSolMgr<double, MV, OP>(
                rcp(&problem, false),
                rcp(&belosList, false)
                ));
      } else {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Unknown solver type \"" << solver << "\".");
      }

      // Perform solve
      RCP<Teuchos::Time> solveTime =
        Teuchos::TimeMonitor::getNewTimer("Linear system solve");
      {
        Teuchos::TimeMonitor tm(*solveTime);
        Belos::ReturnType ret = newSolver->solve();
        success = ret == Belos::Converged;
      }

//         *out << newSolver->getNumIters() << std::endl;
//         // Compute actual residuals.
//         bool badRes = false;
//         Teuchos::Array<double> actual_resids(1);
//         Teuchos::Array<double> rhs_norm(1);
//         Epetra_Vector resid(keoMatrix->OperatorRangeMap());
//         OPT::Apply(*keoMatrix, *x, resid);
//         MVT::MvAddMv(-1.0, resid, 1.0, *b, resid);
//         MVT::MvNorm(resid, actual_resids);
//         MVT::MvNorm(*b, rhs_norm);
//         if (proc_verbose) {
//           std::cout<< "---------- Actual Residuals (normalized) ----------" <<std::endl<<std::endl;
//           for (int i=0; i<1; i++) {
//             double actRes = actual_resids[i]/rhs_norm[i];
//             std::cout << "Problem " << i << " : \t" << actRes << std::endl;
//             if (actRes > 1.0e-10) badRes = true;
//           }
//         }
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, *out, success);

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
// =========================================================================
RCP<Epetra_CrsMatrix>
contructEpetraMatrix(const int n,
                     const Epetra_Comm & eComm)
{
  RCP<Epetra_CrsMatrix> A;
  // Build the matrix (-1,2,-1).
  Epetra_Map map(n, 0, eComm);
  int * myGlobalElements = map.MyGlobalElements();
  A = rcp(new Epetra_CrsMatrix(Copy, map, 3));
  double vals[] = {-1.0, 2.0, -1.0};
  for (int k = 0; k < map.NumMyElements(); k++) {
    if (myGlobalElements[k] == 0) {
      int cols[] = {myGlobalElements[k], myGlobalElements[k]+1};
      TEUCHOS_ASSERT_EQUALITY(0, A->InsertGlobalValues(myGlobalElements[k],
                              2,
                              &vals[1],
                              cols));
    } else if (myGlobalElements[k] == n-1) {
      int cols[] = {myGlobalElements[k]-1, myGlobalElements[k]};
      TEUCHOS_ASSERT_EQUALITY(0, A->InsertGlobalValues(myGlobalElements[k],
                              2,
                              vals,
                              cols));
    } else {
      int cols[] = {myGlobalElements[k]-1, myGlobalElements[k], myGlobalElements[k]+1};
      TEUCHOS_ASSERT_EQUALITY(0, A->InsertGlobalValues(myGlobalElements[k],
                              3,
                              vals,
                              cols));
    }
  }
  TEUCHOS_ASSERT_EQUALITY(0, A->FillComplete(true));

  return A;
}
// =========================================================================
