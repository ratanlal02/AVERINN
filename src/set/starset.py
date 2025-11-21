"""
Author: Ratan Lal
Date : November 4, 2023
"""
from abc import ABC
from typing import Tuple

import numpy.typing as npt
import numpy as np

from src.set.set import Set
from src.solver.gurobi import Gurobi
from src.solver.solver import Solver


class StarSet(Set, ABC):
    """
    The class StarSet capture both convex/non-convex set
    Star set is expressed by x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n] = V*b,
    where V = [c, v[1], v[2], ..., v[n]], where c, v[1],..., v[n] is a column vector
    and b =[1, a[1], a[2], ... , a[n]]^{T}, C*a <=d, constraints on a[i]
    """

    def __init__(self, matBasisV: npt.ArrayLike, matConstraintC: npt.ArrayLike,
                 arrayConstraintd: npt.ArrayLike, arrayPredicateLow: npt.ArrayLike = None,
                 arrayPredicateHigh: npt.ArrayLike = None, arrayStateLow: npt.ArrayLike = None,
                arrayStateHigh: npt.ArrayLike = None):
        """
        Initialize an instance of StarSet
        :param matBasisV: a matrix where first column captures center and rest of them are vertices
        :type matBasisV: npt.ArrayLike
        :param matConstraintC: a matrix for the constraint Ca<=d
        :type matConstraintC: npt.ArrayLike
        :param arrayConstraintd: one dimensional array d for the constraint Ca<=d
        :type arrayConstraintd: npt.ArrayLike
        :param arrayPredicateLow: one dimensional array for the lower bound on a[i]s
        :type arrayPredicateLow: npt.ArrayLike
        :param arrayPredicateHigh: one dimensional array for the upper bound on a[i]s
        :type arrayPredicateHigh: npt.ArrayLike
        :param arrayStateLow: one dimensional array for the lower bound on the state
        :type arrayStateLow: npt.ArrayLike
        :param arrayStateHigh: one dimensional array for the upper bound on the state
        :type arrayStateHigh: npt.ArrayLike
        """
        # Matrix for center and vertices
        self.__matBasisV__: npt.ArrayLike = matBasisV

        # Matrix C for constraint Ca<=d
        self.__matConstraintC__: npt.ArrayLike = matConstraintC

        # A dimensional array for d in constraint Ca<=d
        self.__arrayConstraintd__: npt.ArrayLike = arrayConstraintd

        # A one dimensional array for the lower bound on the predicate variables
        self.__arrayPredicateLow__: npt.ArrayLike = arrayPredicateLow

        # A one dimensional array for the upper bound on the predicate variables
        self.__arrayPredicateHigh__: npt.ArrayLike = arrayPredicateHigh

        # A one dimensional array for the lower bound on the state variables
        self.__arrayStateLow__: npt.ArrayLike = arrayStateLow

        # A one dimensional array for the upper bound on the state variables
        self.__arrayStateHigh__: npt.ArrayLike = arrayStateHigh

        # Number of state variables
        self.intDim: int = matBasisV.shape[0]

        # Number of predicate variables
        self.intNumVar: int = matConstraintC.shape[1]

    ############################################
    ########## Methods for attributes  #########
    ############################################
    def getMatBasisV(self) -> npt.ArrayLike:
        """
        Return Basis interval matrix
        :return: (objIMBasisV -> IntervalMatrix )
        """
        return self.__matBasisV__

    def getMatConstraintC(self) -> npt.ArrayLike:
        """
        Return matrix C in the predicate constraints C * alpha <= d
        :return: (matConstraintC -> npt.ArrayLike )
        """
        return self.__matConstraintC__

    def getArrayConstraintd(self) -> npt.ArrayLike:
        """
        Return array d in the predicate constraints C * alpha <= d
        :return: (arrayConstraintd -> npt.ArrayLike )
        """
        return self.__arrayConstraintd__

    def getArrayPredicateLow(self) -> npt.ArrayLike:
        """
        Return lower bound  of alpha in the predicate C * alpha <= d
        :return: (arrayPredicateLow -> npt.ArrayLike)
        """
        return self.__arrayPredicateLow__

    def getArrayPredicateHigh(self) -> npt.ArrayLike:
        """
        Return upper bound  of alpha in the predicate C * alpha <= d
        :return: (arrayPredicateHigh -> npt.ArrayLike)
        """
        return self.__arrayPredicateHigh__

    def getArrayStateLow(self) -> npt.ArrayLike:
        """
        Return lower bound  of state x in x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n]
        :return: (arrayStateLow -> npt.ArrayLike)
        """
        return self.__arrayStateLow__

    def getArrayStateHigh(self) -> npt.ArrayLike:
        """
        Return upper bound of state x in  x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n]
        :return: (arrayStateHigh -> npt.ArrayLike)
        """
        return self.__arrayStateHigh__

    ############################################
    ########## Methods for only SS     #########
    ############################################
    def getNumOfPredVars(self) -> int:
        """
        Return the number of predicate variables
        :return: (intPredVars -> int)
        """
        return self.__matConstraintC__.shape[1]

    def getNumOfPredicates(self) -> int:
        """
        Return the number of predicates
        :return: (intPredicates -> int)
        """
        return self.__matConstraintC__.shape[0]

    ############################################
    ########## Common Methods  ################
    ############################################

    def linearMap(self, matLow: npt.ArrayLike, matHigh:npt.ArrayLike = None) -> Set:
        """
        Compute affine map without input of an interval star set
        ::param matLow: lower matrix of an interval matrix for weight
        :type matLow: npt.ArrayLike
        :param matHigh: upper matrix of an interval matrix for weight
        :type matHigh: npt.ArrayLike
        :return: (objSS -> 'StarSet') an instance of a star set
        """
        # matrix multiplication
        matbasisnewV: npt.ArrayLike = np.array(np.matmul(matLow, self.__matBasisV__))

        # create new star set
        objStarSet: Set = StarSet(matbasisnewV, self.__matConstraintC__, self.__arrayConstraintd__)

        # return affine map without input of a star set
        return objStarSet

    def affineMap(self, matLow: npt.ArrayLike, arrayLow: npt.ArrayLike, matHigh: npt.ArrayLike = None,
                  arrayHigh: npt.ArrayLike = None) -> Set:
        """
        Affine mapping of the star set
        :param matLow: lower matrix an interval matrix for weight
        :type matLow: npt.ArrayLike
        :param matHigh: upper matrix an interval matrix for weight
        :type matHigh: npt.ArrayLike
        :param arrayLow: lower array of an interval array for bias
        :type arrayLow: npt.ArrayLike
        :param arrayHigh: upper array of an interval array for bias
        :type arrayHigh: npt.ArrayLike
        :return: (objSS -> StarSet) an instance of StarSet representing WX+b,
                where X is a self
        """
        # Matrix multiplication of matW with center and vertices of X
        matBasisNewV: npt.ArrayLike = np.array(np.matmul(matLow, self.__matBasisV__))

        # shift the center, c' = c' + b
        matBasisNewV[:, 0] = matBasisNewV[:, 0] + arrayLow

        # Create affine map of X
        objStarSet: Set = StarSet(matBasisNewV, self.__matConstraintC__, self.__arrayConstraintd__)

        # Return affine StarSet
        return objStarSet

    def intersectPositiveHalfSpaceByIndex(self, intIndex: int) -> Set:
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]>=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objIPHSStarSet -> StarSet) intersection between self and x[intIndex]>=0
        """
        # Compute predicate matrix for the intersection
        matConstraintIPHSC: npt.ArrayLike = np.concatenate((self.__matConstraintC__,
                                                            np.array([-self.__matBasisV__[intIndex, 1:]])), axis=0)

        # Compute column vector for the predicate constraint
        arrayConstraintIPHSd: npt.ArrayLike = np.concatenate((self.__arrayConstraintd__,
                                                             np.array([[self.__matBasisV__[intIndex][0]]])), axis=0)

        # Create star set for the intersection
        objIPHSStarSet: Set = StarSet(self.__matBasisV__, matConstraintIPHSC, arrayConstraintIPHSd)

        # Return the star set for the intersection
        return objIPHSStarSet

    def intersectNegativeHalfSpaceByIndex(self, intIndex: int) -> Set:
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]<=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objIPHSStarSet -> StarSet) intersection between self and x[intIndex]>=0
        """
        # Compute predicate matrix for the intersection
        matConstraintINHSC: npt.ArrayLike = np.concatenate((self.__matConstraintC__,
                                                            np.array([self.__matBasisV__[intIndex, 1:]])), axis=0)

        # Compute column vector for the predicate constraint
        arrayConstraintINHSd: npt.ArrayLike = np.concatenate((self.__arrayConstraintd__,
                                                             np.array([[-self.__matBasisV__[intIndex][0]]])), axis=0)

        # Create star set for the intersection
        objINHSStarSet: Set = StarSet(self.__matBasisV__, matConstraintINHSC, arrayConstraintINHSd)

        # Return the star set for the intersection
        return objINHSStarSet

    def minkowskiSum(self, objStarSet: Set) -> Set:
        """
        Minkowski sum of two star sets
        :param objStarSet: second star set
        :type objStarSet: StarSet
        :return: (objSumStarSet -> StarSet) minkowski sum of two star sets
        """
        # Horizontal concatenation of two matrices
        matBasisSumV: npt.ArrayLike = np.concatenate((self.__matBasisV__, objStarSet.getMatBasisV()[:, 1:]), axis=1)

        # Sum of centers
        matBasisSumV[:, 0] = self.__matBasisV__[:, 0] + objStarSet.getMatBasisV()[:, 0]

        # Express matConstraintC of both self and objStarSet as a block diagonal matrix
        r1: int = np.shape(self.__matConstraintC__)[0]
        c1: int = np.shape(self.__matConstraintC__)[1]
        r2: int = np.shape(objStarSet.getMatConstraintC())[0]
        c2: int = np.shape(objStarSet.getMatConstraintC())[1]
        matConstraintC: npt.ArrayLike = np.block([[self.__matConstraintC__, np.zeros((r1, c2))],
                                                  [np.zeros((r2, c1)), objStarSet.matConstraintC]])

        # Vertical concatenation of column vectors cvecConstraintd of both self and objStarSet
        arrayConstraintd: npt.ArrayLike = np.concatenate((self.__arrayConstraintd__, objStarSet.getArrayConstraintd()), axis=0)

        # Create an instance of StarSet for the minkowski sum
        objSumStarSet: Set = StarSet(matBasisSumV, matConstraintC, arrayConstraintd)

        # Return minkowski sum of two star sets
        return objSumStarSet

    def getRange(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Return the range of the IntervalStarSet
        """
        # Encode Interval Star Set in Gurobi
        grbModel, dictVars = self.__encode__()
        objSolver: Solver = Gurobi(grbModel, dictVars)
        objSet: Set = objSolver.outputRange()
        rangeISS: Tuple[npt.ArrayLike, npt.ArrayLike] = (objSet.getArrayLow(), objSet.getArrayHigh())
        return rangeISS



    def intersectHalfSpace(self, matH: npt.ArrayLike, cvecg: npt.ArrayLike) -> 'StarSet':
        """
        Compute intersection between star set and half space expressed by
        Hx<=g
        :param matH: matrix for the half space
        :type matH: npt.ArrayLike
        :param cvecg: column vector for the half space
        :type cvecg: npt.ArrayLike
        :return: (objIHSStarSet -> StarSet) star set for the intersection between self and Hx<=g
        """
        # Compute predicate matrix for the intersection StarSet
        C: npt.ArrayLike = np.matmul(matH, self.matBasisV[:, 1:])

        # Vertical concatenation
        matConstraintIHSC: npt.ArrayLike = np.concatenate((self.matConstraintC, C), axis=0)

        # Compute column vector for the predicate
        cvecg = cvecg - np.matmul(matH, self.matBasisV[:, 0])

        # Compute column vector for the constraint
        cvecConstraintIHSd: npt.ArrayLike = np.concatenate((self.cvecConstraintd, cvecg), axis=0)

        # Create star set for the intersection with the half space
        objIHSStarSet: StarSet = StarSet(self.matBasisV, matConstraintIHSC, cvecConstraintIHSd)

        # Return star set for the intersection with the half space
        return objIHSStarSet



