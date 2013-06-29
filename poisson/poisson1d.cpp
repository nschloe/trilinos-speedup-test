#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

// Epetra includes.
#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>

// Tpetra includes.
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>

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
typedef Belos::MultiVecTraits<ST,MV>     MVT;
typedef Belos::OperatorTraits<ST,MV,OP>  OPT;

enum Operator { JAC, KEO, KEOREG, POISSON1D };
// =============================================================================
int main (int argc, char *argv[])
{
    // Create a communicator for Epetra objects
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    Epetra_MpiComm eComm(MPI_COMM_WORLD);
#else
    Epetra_SerialComm eComm();
#endif

    Tpetra::DefaultPlatform::DefaultPlatformType &platform =
      Tpetra::DefaultPlatform::getDefaultPlatform();
    RCP<const Teuchos::Comm<int> > tComm = platform.getComm();

    typedef Tpetra::DefaultPlatform::DefaultPlatformType::NodeType Node;
    typedef Tpetra::Map<int,int,Node>                      Map;

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
      RCP<Epetra_CrsMatrix> epetra_A;
      {
          Teuchos::TimeMonitor tm(*matrixConstructTime);
          // Build the matrix (-1,2,-1).
          Epetra_Map map(n, 0, eComm);
          int * myGlobalElements = map.MyGlobalElements();
          epetra_A = rcp(new Epetra_CrsMatrix(Copy, map, 3));
          double vals[] = {-1.0, 2.0, -1.0};
          for (int k=0; k < map.NumMyElements(); k++)
          {
            if (myGlobalElements[k] == 0)
            {
              int cols[] = {myGlobalElements[k], myGlobalElements[k]+1};
              TEUCHOS_ASSERT_EQUALITY(0, epetra_A->InsertGlobalValues(myGlobalElements[k],
                                                                      2,
                                                                      &vals[1],
                                                                      cols));
            }
            else if (myGlobalElements[k] == n-1)
            {
              int cols[] = {myGlobalElements[k]-1, myGlobalElements[k]};
              TEUCHOS_ASSERT_EQUALITY(0, epetra_A->InsertGlobalValues(myGlobalElements[k],
                                                                      2,
                                                                      vals,
                                                                      cols));
            }
            else
            {
              int cols[] = {myGlobalElements[k]-1, myGlobalElements[k], myGlobalElements[k]+1};
              TEUCHOS_ASSERT_EQUALITY(0, epetra_A->InsertGlobalValues(myGlobalElements[k],
                                                                      3,
                                                                      vals,
                                                                      cols));
            }
          }
          TEUCHOS_ASSERT_EQUALITY(0, epetra_A->FillComplete(true));
      }
//       epetra_A->Print(std::cout);

      // Construct Tpetra matrix.
      RCP<Teuchos::Time> tpetraMatrixConstructTime =
          Teuchos::TimeMonitor::getNewTimer("Tpetra matrix construction");
      RCP<Tpetra::CrsMatrix<double,int> > tpetra_A;
      {
        Teuchos::TimeMonitor tm(*tpetraMatrixConstructTime);
        RCP<const Tpetra::Map<int> > map =
          Tpetra::createUniformContigMap<int,int>(n, tComm);
        // Get update list and number of local equations from newly created map.
        const size_t numMyElements = map->getNodeNumElements();
        Teuchos::ArrayView<const int> myGlobalElements = map->getNodeElementList();
        // Create a CrsMatrix using the map, with a dynamic allocation of 3 entries per row
        tpetra_A = Tpetra::createCrsMatrix<double>(map, 3);
        // Add rows one-at-a-time
        for (size_t i=0; i<numMyElements; i++)
        {
          if (myGlobalElements[i] == 0)
          {
            tpetra_A->insertGlobalValues(myGlobalElements[i],
                                         Teuchos::tuple<int>(myGlobalElements[i], myGlobalElements[i]+1),
                                         Teuchos::tuple<double> (2.0, -1.0));
          }
          else if (myGlobalElements[i] == n-1)
          {
            tpetra_A->insertGlobalValues(myGlobalElements[i],
                                         Teuchos::tuple<int>(myGlobalElements[i]-1, myGlobalElements[i]),
                                         Teuchos::tuple<double> (-1.0, 2.0));
          }
          else {
          tpetra_A->insertGlobalValues(myGlobalElements[i],
                                       Teuchos::tuple<int>(myGlobalElements[i]-1, myGlobalElements[i], myGlobalElements[i]+1),
                                       Teuchos::tuple<double> (-1.0, 2.0, -1.0));
          }
        }
        // Complete the fill, ask that storage be reallocated and optimized
        tpetra_A->fillComplete();
        //tpetra_A->fillComplete(Tpetra::DoOptimizeStorage);
      }

//       RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(rcpFromRef(std::cout));
//       tpetra_A->describe(*fos, Teuchos::VERB_EXTREME);
//       std::cout << std::endl << tpetra_A->description() << std::endl << std::endl;


      // create initial guess and right-hand side
      RCP<Epetra_Vector> epetra_x =
        rcp(new Epetra_Vector(epetra_A->OperatorDomainMap()));
      RCP<Epetra_MultiVector> epetra_b =
        rcp(new Epetra_Vector(epetra_A->OperatorRangeMap()));
      // epetra_b->Random();
      TEUCHOS_ASSERT_EQUALITY(0, epetra_b->PutScalar(1.0));

      // create tpetra vectors
      RCP<Tpetra::Vector<double,int> > tpetra_x =
        rcp(new Tpetra::Vector<double,int>(tpetra_A->getDomainMap()));
      RCP<Tpetra::Vector<double,int> > tpetra_b =
        rcp(new Tpetra::Vector<double,int>(tpetra_A->getRangeMap()));
      tpetra_b->putScalar(1.0);

      if (action.compare("matvec") == 0)
      {
        TEUCHOS_ASSERT_EQUALITY(0, epetra_x->PutScalar(1.0));
        RCP<Teuchos::Time> mvTime = Teuchos::TimeMonitor::getNewTimer("Epetra operator apply");
        {
          Teuchos::TimeMonitor tm(*mvTime);
          // Don't TEUCHOS_ASSERT_EQUALITY() here for speed.
          epetra_A->Apply(*epetra_x, *epetra_b);
        }

        tpetra_x->putScalar(1.0);
        RCP<Teuchos::Time> tmvTime = Teuchos::TimeMonitor::getNewTimer("Tpetra operator apply");
        {
          Teuchos::TimeMonitor tm(*tmvTime);
          tpetra_A->apply(*tpetra_x, *tpetra_b);
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
        }
        else
          belosList.set("Verbosity", Belos::Errors + Belos::Warnings);

        belosList.set("Output Style", (int)Belos::Brief); // Belos::General, Belos::Brief
        belosList.set("Maximum Iterations", 1000);

        // Construct an unpreconditioned linear problem instance.
        Belos::LinearProblem<double,MV,OP> problem(epetra_A, epetra_x, epetra_b);
        bool set = problem.setProblem();
        TEUCHOS_TEST_FOR_EXCEPTION(!set,
                                   std::logic_error,
                                   "ERROR:  Belos::LinearProblem failed to set up correctly!");
        // -----------------------------------------------------------------------
        // Create an iterative solver manager.
        RCP<Belos::SolverManager<double,MV,OP> > newSolver;
        if (action.compare("solve_cg") == 0)
        {
          belosList.set("Assert Positive Definiteness", false);
          newSolver =
            rcp(new Belos::PseudoBlockCGSolMgr<double,MV,OP>(rcp(&problem,false),
                                                             rcp(&belosList,false)
                                                             ));
        }
        else if (action.compare("solve_minres") == 0)
        {
          newSolver =
            rcp(new Belos::MinresSolMgr<double,MV,OP>(rcp(&problem,false),
                                                      rcp(&belosList,false)
                                                      ));
        }
        else if (action.compare("solve_gmres") == 0)
        {
          newSolver =
            rcp(new Belos::PseudoBlockGmresSolMgr<double,MV,OP>(rcp(&problem,false),
                                                                rcp(&belosList,false)
                                                                ));
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Unknown solver type \"" << solver << "\".");
        }

        // Perform solve
        RCP<Teuchos::Time> solveTime =
          Teuchos::TimeMonitor::getNewTimer("Linear system solve");
        {
            Teuchos::TimeMonitor tm(*solveTime);
            Belos::ReturnType ret = newSolver->solve();
            success = ret==Belos::Converged;
        }

//         *out << newSolver->getNumIters() << std::endl;
//         // Compute actual residuals.
//         bool badRes = false;
//         Teuchos::Array<double> actual_resids(1);
//         Teuchos::Array<double> rhs_norm(1);
//         Epetra_Vector resid(keoMatrix->OperatorRangeMap());
//         OPT::Apply(*keoMatrix, *epetra_x, resid);
//         MVT::MvAddMv(-1.0, resid, 1.0, *epetra_b, resid);
//         MVT::MvNorm(resid, actual_resids);
//         MVT::MvNorm(*epetra_b, rhs_norm);
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

#ifdef HAVE_MPI
      MPI_Finalize();
#endif

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
// =========================================================================
