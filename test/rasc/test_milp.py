import copy
from typing import List, Tuple, Dict
from unittest import TestCase

from gurobipy import Model, GRB, Var

from src.gnn.gnn import GNN
from src.parser.parser import Parser
from src.parser.sherlock import Sherlock
import numpy as np
import numpy.typing as npt

from src.rasc.milp import Milp
from src.rasc.technique import Technique
from src.set.box import Box
from src.set.grbset import GRBSet
from src.set.set import Set
from src.types.lastrelu import LastRelu
from src.types.solvertype import SolverType
from src.types.techniquetype import TechniqueType
from src.gnn.gnnuts import GNNUTS


class TestMilp(TestCase):
    def test_reach_set(self):
        """
        Test reachset function
        :return: None
        """
        ##################################
        ########## Parameters  ###########
        ##################################
        solverType: SolverType = SolverType.Gurobi
        lastRelu: LastRelu = LastRelu.NO
        file = "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/test/nn.txt"
        # Create NeuralNetwork
        objParser: Parser = Sherlock(file)
        objGNN: GNN = objParser.getNetwork()
        #########################################
        ########### Input Specification #########
        #########################################
        arrayLow: npt.ArrayLike = np.array([1], dtype='f')
        arrayHigh: npt.ArrayLike = np.array([1], dtype='f')
        objInputSet: Set = Box(arrayLow, arrayHigh)

        # Compute Reach set
        objTechnique: Technique = Milp(objGNN, objInputSet, None, solverType, lastRelu)
        listSets: List[Set] = objTechnique.reachSet()

        arrayLow: npt.ArrayLike = listSets[0].getLowerBound()
        arrayHigh: npt.ArrayLike = listSets[0].getUpperBound()
        np.testing.assert_array_equal(arrayLow, [1], "Reach set are not the same")
        np.testing.assert_array_equal(arrayHigh, [1], "Reach set are not the same")

    def test_check_satisfy(self):
        """
        `Check there is an instance reaching unsafe set
        :return: None
        """
        ##################################
        ########## Parameters  ###########
        ##################################
        solverType: SolverType = SolverType.Gurobi
        reachType: TechniqueType = TechniqueType.MILP
        lastRelu: LastRelu = LastRelu.NO
        file = "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/test/nn.txt"
        # Create NeuralNetwork
        objParser: Parser = Sherlock(file)
        objGNN: GNN = objParser.getNetwork()
        #########################################
        ########### Input Specification #########
        #########################################
        arrayLow: npt.ArrayLike = np.array([1], dtype='f')
        arrayHigh: npt.ArrayLike = np.array([1], dtype='f')
        objInputSet: Set = Box(arrayLow, arrayHigh)
        #########################################
        ########### Output Specification #########
        #########################################
        dictVars: Dict[int, Dict[int, Var]] = {}
        dictVars[1] = {}
        objModel: Model = Model()
        dictVars[1][1] = objModel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                            name='x_' + str(1) + '_' + str(1))
        objModel.addConstr(dictVars[1][1] >= 2)
        objModel.update()
        objSet: Set = GRBSet(objModel, dictVars)
        # Check safety against unsafe set
        objTechnique: Technique = Milp(objGNN, objInputSet, None, solverType, lastRelu)
        isSatisfy: bool = objTechnique.checkSatisfy(objSet)
        self.assertEqual(isSatisfy, False, "There is no instance satisfy unsafe set")

    def test_check_safety(self):
        """
        Check safety against unsafe set
        :return: None
        """
        ##################################
        ########## Parameters  ###########
        ##################################
        solverType: SolverType = SolverType.Gurobi
        lastRelu: LastRelu = LastRelu.NO
        file = "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/test/nn.txt"
        # Create NeuralNetwork
        objParser: Parser = Sherlock(file)
        objGNN: GNN = objParser.getNetwork()
        #########################################
        ########### Input Specification #########
        #########################################
        arrayLow: npt.ArrayLike = np.array([1], dtype='f')
        arrayHigh: npt.ArrayLike = np.array([1], dtype='f')
        objInputSet: Set = Box(arrayLow, arrayHigh)
        #########################################
        ########### Output Specification #########
        #########################################
        A: npt.ArrayLike = np.array([[-1]], dtype='f')
        b: npt.ArrayLike = np.array([-2], dtype='f')
        outputConstr: Tuple[npt.ArrayLike, npt.ArrayLike] = ([A], [b])
        # Check safety against unsafe set
        objTechnique: Technique = Milp(objGNN, objInputSet, outputConstr, solverType, lastRelu)
        isSafe: bool = objTechnique.checkSafety()
        self.assertEqual(isSafe, True, "There is an instance satisfy unsafe set")

