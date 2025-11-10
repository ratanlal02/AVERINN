"""
Author: Ratan Lal
Date : January 28, 2025
"""
from typing import List, Dict

from gurobipy import Model, GRB, quicksum, Var

from src.set.box import Box
from src.set.intervalmatrix import IntervalMatrix
from src.set.intervalstarset import IntervalStarSet
from src.set.set import Set
import numpy.typing as npt
import numpy as np

from src.solver.gurobi import Gurobi
from src.solver.solver import Solver
from src.types.datatype import DataType
from src.types.solvertype import SolverType
from src.utilities.log import Log


class SetUTS:
    """
    Common functionality related to Set
    """

    @staticmethod
    def displayLinConstr(A: npt.ArrayLike, b: npt.ArrayLike):
        """
        Print Ax <= b in the form of A b
        :param A: Two dimensional array
        :type A: npt.ArrayLike
        :param b: One dimensional array
        :type b: npt.ArrayLike
        :return: None
        """
        for row in range(len(A)):
            paddedRows = []
            paddedRows.append("         " + str(A[row]))
            paddedRows.append(str(b[row]))
            Log.message(str('     '.join(paddedRows)) + "\n")

    @staticmethod
    def displayDictOfDictOfSets(dictPartition: Dict[int, Dict[int, set]]):
        """
        Print dictionary of dictionaries of sets
        :param dictPartition: dictionary of dictionaries of sets
        :type dictPartition: Dict[int, Dict[int, Set]]
        :return: None
        """
        for intLayer in dictPartition.keys():
            paddedRows = []
            paddedRows.append("       " + str(intLayer) + " : " + str(dictPartition[intLayer]))
            Log.message(str('     '.join(paddedRows)) + "\n")

    @staticmethod
    def displayDictOfDictOfValues(dictCE: Dict[int, Dict[int, DataType.RealType]]):
        """
        Print dictionary of dictionaries of floats
        :param dictPartition: dictionary of dictionaries of floats
        :type dictPartition: Dict[int, Dict[int, float]]
        :return: None
        """
        for intLayer in dictCE.keys():
            paddedRows = []
            paddedRows.append("         " + str(intLayer) + " : " + str(dictCE[intLayer]))
            Log.message(str('     '.join(paddedRows)) + "\n")

    @staticmethod
    def displayListOfSets(listOfSets: List[Set]):
        """
        Print list of Sets
        :param listOfSets: list of Sets
        :type listOfSets: List[Set]
        :return: None
        """
        i: int = 0
        for objSet in listOfSets:
            Log.message("           Set " + str(i + 1) + "\n")
            i += 1
            Log.message(objSet.display())

    @staticmethod
    def rangeOfSets(listOfSets: List[Set]) -> Set:
        """
        Compute an over-approximation of sets as a set
        :param listOfSets: List[Set]
        :return: (listOfSet -> List[Set])
        """
        intDim: int = listOfSets[0].getDimension()
        arrayLow: npt.ArrayLike = [min([objSet.getRange()[0][i] for objSet in listOfSets]) for i in range(intDim)]
        arrayHigh: npt.ArrayLike = [max([objSet.getRange()[1][i] for objSet in listOfSets]) for i in range(intDim)]

        objSet: Set = Box(arrayLow, arrayHigh)

        return objSet

    @staticmethod
    def toIntervalStarSet(objSet: Set) -> Set:
        """
        Convert a given get into IntervalStarSet as a Set
        """
        rangeSet = objSet.getRange()
        arrayLow = rangeSet[0]
        arrayHigh = rangeSet[1]
        intDim = objSet.getDimension()
        # define matBasisV, that is, center point and vertices
        matBasisVLow = []
        for i in range(intDim):
            temp = [0.0 for j in range(intDim + 1)]
            temp[0] = (arrayLow[i] + arrayHigh[i]) / 2
            temp[i + 1] = (arrayHigh[i] - arrayLow[i]) / 2
            matBasisVLow.append(temp)
        matBasisVLow = np.array(matBasisVLow, dtype=np.float64)
        objIMBasisV: IntervalMatrix = IntervalMatrix(matBasisVLow, matBasisVLow)
        # define constraint matrix for -1 <=a[1]<=1, ....
        matConstraintC = []
        for i in range(intDim):
            temp = [0.0 for j in range(intDim)]
            temp[i] = 1
            matConstraintC.append(temp)
            temp = [0 for j in range(intDim)]
            temp[i] = -1
            matConstraintC.append(temp)
        matConstraintC = np.array(matConstraintC)

        arrayConstraintd = np.array([1 for j in range(2 * intDim)])

        objSet: Set = IntervalStarSet(objIMBasisV, matConstraintC, arrayConstraintd)
        return objSet
