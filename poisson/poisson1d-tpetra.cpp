#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

// Tpetra includes.
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>

//#include "BelosConfigDefs.hpp"
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
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
//typedef double                           ST;
//typedef Tpetra::MultiVector<double,int,int>  MV;
//typedef Tpetra::Operator<double,int>     OP;
//typedef Belos::MultiVecTraits<ST,MV>     MVT;
//typedef Belos::OperatorTraits<ST,MV,OP>  OPT;

// Set up Tpetra typedefs.
typedef double scalar_type;
typedef int local_ordinal_type;
typedef long global_ordinal_type;
typedef Kokkos::DefaultNode::DefaultNodeType node_type;
typedef Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> vector_type;
typedef Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type> matrix_type;
typedef Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type, node_type> op_type;

enum Operator { JAC, KEO, KEOREG, POISSON1D };
// =============================================================================
RCP<matrix_type>
contructTpetraMatrix(const int n, const RCP<const Teuchos::Comm<int> > & comm);
// =============================================================================
int main (int argc, char *argv[])
{
    Teuchos::oblackholestream blackHole;
    Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackHole);
    RCP<const Teuchos::Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

    const RCP<Teuchos::FancyOStream> out =
        Teuchos::VerboseObjectBase::getDefaultOStream();

    bool success = true;
    try
    {
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
      // Construct Tpetra matrix.
      RCP<Teuchos::Time> tpetraMatrixConstructTime =
          Teuchos::TimeMonitor::getNewTimer("Tpetra matrix construction");
      RCP<matrix_type> A;
      {
      Teuchos::TimeMonitor tm(*tpetraMatrixConstructTime);
      A = contructTpetraMatrix(n, comm);
      }
//       RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(rcpFromRef(std::cout));
//       A->describe(*fos, Teuchos::VERB_EXTREME);
//       std::cout << std::endl << A->description() << std::endl << std::endl;

      // create tpetra vectors
      RCP<vector_type> x = rcp(new vector_type(A->getDomainMap()));
      RCP<vector_type> b = rcp(new vector_type(A->getRangeMap()));
      b->putScalar(1.0);

      if (action.compare("matvec") == 0)
      {
        x->putScalar(1.0);
        RCP<Teuchos::Time> tmvTime = Teuchos::TimeMonitor::getNewTimer("Tpetra operator apply");
        {
        Teuchos::TimeMonitor tm(*tmvTime);
        A->apply(*x, *b);
        }

        // print timing data
        Teuchos::TimeMonitor::summarize();
      }
      else
      {
        // -----------------------------------------------------------------------
        // Belos part
        Teuchos::ParameterList belosList;
        // Relative convergence tolerance requested
        belosList.set("Convergence Tolerance", 1.0e-12);
        if (verbose)
        {
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
        }
        else
          belosList.set("Verbosity", Belos::Errors + Belos::Warnings);

        belosList.set("Output Style", (int)Belos::Brief); // Belos::General, Belos::Brief
        belosList.set("Maximum Iterations", 1000);

        // Construct an unpreconditioned linear problem instance.
        Belos::LinearProblem<scalar_type,vector_type,op_type> problem(A, x, b);
//        bool set = problem.setProblem();
//        TEUCHOS_TEST_FOR_EXCEPTION(!set,
//                                   std::logic_error,
//                                   "ERROR:  Belos::LinearProblem failed to set up correctly!");
//        // -----------------------------------------------------------------------
//        // Create an iterative solver manager.
//        RCP<Belos::SolverManager<double,MV,OP> > newSolver;
//        if (action.compare("solve_cg") == 0)
//        {
//          belosList.set("Assert Positive Definiteness", false);
//          newSolver =
//            rcp(new Belos::PseudoBlockCGSolMgr<double,MV,OP>(rcp(&problem,false),
//                                                             rcp(&belosList,false)
//                                                             ));
//        }
//        else if (action.compare("solve_minres") == 0)
//        {
//          newSolver =
//            rcp(new Belos::MinresSolMgr<double,MV,OP>(rcp(&problem,false),
//                                                      rcp(&belosList,false)
//                                                      ));
//        }
//        else if (action.compare("solve_gmres") == 0)
//        {
//          newSolver =
//            rcp(new Belos::PseudoBlockGmresSolMgr<double,MV,OP>(rcp(&problem,false),
//                                                                rcp(&belosList,false)
//                                                                ));
//        }
//        else
//        {
//          TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Unknown solver type \"" << solver << "\".");
//        }
//
//        // Perform solve
//        RCP<Teuchos::Time> solveTime =
//          Teuchos::TimeMonitor::getNewTimer("Linear system solve");
//        {
//            Teuchos::TimeMonitor tm(*solveTime);
//            Belos::ReturnType ret = newSolver->solve();
//            success = ret==Belos::Converged;
//        }
//
////         *out << newSolver->getNumIters() << std::endl;
////         // Compute actual residuals.
////         bool badRes = false;
////         Teuchos::Array<double> actual_resids(1);
////         Teuchos::Array<double> rhs_norm(1);
////         Epetra_Vector resid(keoMatrix->OperatorRangeMap());
////         OPT::Apply(*keoMatrix, *epetra_x, resid);
////         MVT::MvAddMv(-1.0, resid, 1.0, *epetra_b, resid);
////         MVT::MvNorm(resid, actual_resids);
////         MVT::MvNorm(*epetra_b, rhs_norm);
////         if (proc_verbose) {
////           std::cout<< "---------- Actual Residuals (normalized) ----------" <<std::endl<<std::endl;
////           for (int i=0; i<1; i++) {
////             double actRes = actual_resids[i]/rhs_norm[i];
////             std::cout << "Problem " << i << " : \t" << actRes << std::endl;
////             if (actRes > 1.0e-10) badRes = true;
////           }
////         }
      }
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS(true, *out, success);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
// =========================================================================
RCP<matrix_type>
contructTpetraMatrix(const int n,
                     const RCP<const Teuchos::Comm<int> > & comm)
{
  RCP<matrix_type> A;
  RCP<const Tpetra::Map<local_ordinal_type,global_ordinal_type> > map =
    Tpetra::createUniformContigMap<local_ordinal_type,global_ordinal_type>(n, comm);
  // Get update list and number of local equations from newly created map.
  const size_t numMyElements = map->getNodeNumElements();
  Teuchos::ArrayView<const global_ordinal_type> myGlobalElements = map->getNodeElementList();
  // Create a CrsMatrix using the map, with a dynamic allocation of 3 entries per row
  A = Tpetra::createCrsMatrix<scalar_type>(map, 3);
  // Add rows one-at-a-time
  for (size_t i=0; i<numMyElements; i++)
  {
    if (myGlobalElements[i] == 0)
    {
      A->insertGlobalValues(myGlobalElements[i],
                                   Teuchos::tuple<global_ordinal_type>(myGlobalElements[i], myGlobalElements[i]+1),
                                   Teuchos::tuple<scalar_type> (2.0, -1.0));
    }
    else if (myGlobalElements[i] == n-1)
    {
      A->insertGlobalValues(myGlobalElements[i],
                                   Teuchos::tuple<global_ordinal_type>(myGlobalElements[i]-1, myGlobalElements[i]),
                                   Teuchos::tuple<scalar_type> (-1.0, 2.0));
    }
    else {
    A->insertGlobalValues(myGlobalElements[i],
                                 Teuchos::tuple<global_ordinal_type>(myGlobalElements[i]-1, myGlobalElements[i], myGlobalElements[i]+1),
                                 Teuchos::tuple<scalar_type> (-1.0, 2.0, -1.0));
    }
  }
  // Complete the fill, ask that storage be reallocated and optimized
  A->fillComplete();
  //A->fillComplete(Tpetra::DoOptimizeStorage);
  return A;
}
// =========================================================================
