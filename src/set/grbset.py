"""
Author: Ratan Lal
Date : November 18, 2024
"""
from abc import ABC
from typing import Dict, List, Tuple
from gurobipy import Model, Var, GRB
from src.set.box import Box
from src.set.set import Set
import numpy.typing as npt
from src.solver.gurobi import Gurobi
from src.solver.solver import Solver
from src.types.datatype import DataType
from src.types.sign import Sign


class GRBSet(Set, ABC):
    def __init__(self, objModel: Model, dictVars: Dict[int, Dict[int, Var]]):
        """
        Initialize GRB set
        """
        self.__objModel__ = objModel
        self.__dictVars__ = dictVars

    def getLowerBound(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayLow -> npt.ArrayLike)
        lower bound of the set
        """
        intDim: int = self.getDimension()
        arrayLow: npt.ArrayLike = [0.0 for i in range(intDim)]
        return arrayLow

    def getUpperBound(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayHigh -> npt.ArrayLike)
        upper bound of the set
        """
        intDim: int = self.getDimension()
        arrayHigh: npt.ArrayLike = [0.0 for i in range(intDim)]
        return arrayHigh

    def getDimension(self) -> int:
        """
        Returns the dimension of the set
        :return: (intDim -> int)
        dimension of the set
        """
        key: int = [v for v in self.__dictVars__.keys()][0]
        return len(self.__dictVars__[key])

    def getSameSignPartition(self) -> List['Set']:
        """
        Partition a set in such a way that for each sub set,
        values in each dimension will have the same sign
        :return: (listSets -> List[Set]) list of subsets
        """
        intDim: int = self.getDimension()
        arrayLow: npt.ArrayLike = [DataType.RealType(0.0) for i in range(intDim)]
        arrayHigh: npt.ArrayLike = [DataType.RealType(0.0) for i in range(intDim)]
        listSets: List['Set'] = [Box(arrayLow, arrayHigh)]
        return listSets

    def getSign(self) -> List[Sign]:
        """
        Returns the list of Sign instances for all
        the dimensions of the set
        :return: (listSigns -> List[Sign])
        """
        intDim: int = self.getDimension()
        listSign: List[Sign] = [Sign.POS for i in range(intDim)]
        return listSign

    def toAbsolute(self) -> npt.ArrayLike:
        """
        Returns the absolute value of the set
        :return: (arrayPoint -> npt.ArrayLike)
        """
        intDim: int = self.getDimension()
        arrayPoint: npt.ArrayLike = [DataType.RealType(0.0) for i in range(intDim)]
        return arrayPoint

    def isEmpty(self) -> bool:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        objSolver: Solver = Gurobi(self.__objModel__, self.__dictVars__)
        return not (objSolver.satisfy())

    def display(self) -> str:
        """
        Display all the constraints
        :return: None
        """
        strGRBSet: str = ""
        listVars: List[Var] = self.__objModel__.getVars()
        numVars: int = len(listVars)
        for constr in self.__objModel__.getConstrs():
            listCoeff: List[float] = []
            for i in range(numVars):
                listCoeff.append(self.__objModel__.getCoeff(constr, listVars[i]))
            constantTerm = constr.getAttr("rhs")
            isFirst: bool = True
            for i in range(numVars):
                if listCoeff[i] != 0.0:
                    if isFirst:
                        strGRBSet += "          " + str(listVars[i].VarName) + "*" + str(listCoeff[i])
                        isFirst = False
                    else:
                        strGRBSet += " + " + str(listVars[i].VarName) + "*" + str(listCoeff[i])

            if constr.Sense == GRB.LESS_EQUAL:
                strGRBSet += " <= " + str(constantTerm)
            elif constr.Sense == GRB.GREATER_EQUAL:
                strGRBSet += " >= " + str(constantTerm)
            else:
                strGRBSet += " == " + str(constantTerm)
            strGRBSet += "\n"

        return strGRBSet

    def getModelAndDictVars(self) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        return self.__objModel__, self.__dictVars__

    def affineMap(self, matW: npt.ArrayLike, arrayB: npt.ArrayLike) -> Set:
        """
        Affine mapping of the box set
        :param matW: a two-dimensional array for a weight matrix
        :type matW: npt.ArrayLike
        :param arrayB: a one dimensional array
        :type arrayB: npt.ArrayLike
        :return: (objAffineBox -> Box) an instance of box representing WX+b,
        """
        objSet: Set = None

        # Return Affine set
        return objSet

    def linearMap(self, matW: npt.ArrayLike) -> Set:
        """
        Linear mapping of the box set
        :param matW: a two-dimensional array for a weight matrix
        :type matW: npt.ArrayLike
        :return: (objLinearBox -> Box) an instance of Box representing WX,
                where X is a self
        """
        objSet: Set = None

        # Return Affine set
        return objSet

    def convexHull(self, objSet: Set) -> Set:
        """
        Construct an instance of Box representing convex hull of two boxes,
        that is, self and objBox
        :param objSet: an instance of Set
        :type objSet: Set
        :return: (objCHSet -> Set) an instance of Set for convex hull
        """
        objSet: Set = None

        # Return Affine set
        return objSet

    def minkowskiSum(self, objSet: Set) -> Set:
        """
        Construct an instance of Box representing minkowski sum of two boxes,
        that is, self and objBox
        :param objSet: an instance of a set
        :type objSet: Set
        :return: (objMWSet -> Set) an instance of Set for minkowski sum
        """
        objSet: Set = None

        # Return Affine set
        return objSet