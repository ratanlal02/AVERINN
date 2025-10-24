"""
Author: Ratan Lal
Date : November 4, 2023
"""
import numpy.typing as npt
import numpy as np


class StarSet:
    """
    The class StarSet capture both convex/non-convex set
    Star set is expressed by x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n] = V*b,
    where V = [c, v[1], v[2], ..., v[n]], where c, v[1],..., v[n] is a column vector
    and b =[1, a[1], a[2], ... , a[n]]^{T}, C*a <=d, constraints on a[i]
    """

    def __init__(self, matBasisV: npt.ArrayLike, matConstraintC: npt.ArrayLike,
                 cvecConstraintd: npt.ArrayLike, cvecPredicateLb: npt.ArrayLike = None,
                 cvecPredicateUb: npt.ArrayLike = None, cvecStateLb: npt.ArrayLike = None,
                 cvecStateUb: npt.ArrayLike = None):
        """
        Initialize an instance of StarSet
        :param matBasisV: a matrix where first column captures center and rest of them are vertices
        :type matBasisV: npt.ArrayLike
        :param matConstraintC: a matrix for the constraint Ca<=d
        :type matConstraintC: npt.ArrayLike
        :param cvecConstraintd: a column vector d for the constraint Ca<=d
        :type cvecConstraintd: npt.ArrayLike
        :param cvecPredicateLb: a column vector for the lower bound on a[i]s
        :type cvecPredicateLb: npt.ArrayLike
        :param cvecPredicateUb: a column vector for the upper bound on a[i]s
        :type cvecPredicateUb: npt.ArrayLike
        :param cvecStateLb: a column vector for the lower bound on the state
        :type cvecStateLb: npt.ArrayLike
        :param cvecStateUb: a column vector for the upper bound on the state
        :type cvecStateUb: npt.ArrayLike
        """
        # Matrix for center and vertices
        self.matBasisV: npt.ArrayLike = matBasisV

        # Matrix C for constraint Ca<=d
        self.matConstraintC: npt.ArrayLike = matConstraintC

        # A column vector d for constraint Ca<=d
        self.cvecConstraintd: npt.ArrayLike = cvecConstraintd

        # A column vector for the lower bound on the predicate variables
        self.cvecPredicateLb: npt.ArrayLike = cvecPredicateLb

        # A column vector for the upper bound on the predicate variables
        self.cvecPredicateUb: npt.ArrayLike = cvecPredicateUb

        # A column vector for the lower bound on the state variables
        self.cvecStateLb: npt.ArrayLike = cvecStateLb

        # A column vector for the upper bound on the state variables
        self.cvecStateUb: npt.ArrayLike = cvecStateUb

        # Number of state variables
        self.intDim: int = matBasisV.shape[0]

        # Number of predicate variables
        self.intNumVar: int = matConstraintC.shape[1]

    def linearMap(self, matW: npt.ArrayLike) -> 'StarSet':
        """
        Compute affine map without input of a star set
        :param matW: weight matrix
        :type matW: npt.Array for matrix
        :return: (objStarSet -> 'StarSet') an instance of a star set
        """
        # matrix multiplication
        matbasisnewV: npt.ArrayLike = np.array(np.matmul(matW, self.matBasisV))

        # create new star set
        objStarSet: StarSet = StarSet(matbasisnewV, self.matConstraintC, self.cvecConstraintd)

        # return affine map without input of a star set
        return objStarSet

    def affineMap(self, matW: npt.ArrayLike, cvecb: npt.ArrayLike) -> 'StarSet':
        """
        Affine mapping of the star set
        :param matW: a two-dimensional array for a weight matrix
        :type matW: npt.ArrayLike
        :param cvecb: one dimensional array
        :type cvecb: npt.ArrayLike
        :return: (objStarSet -> StarSet) an instance of StarSet representing WX+b,
                where X is a self
        """
        # Matrix multiplication of matW with center and vertices of X
        matBasisNewV: npt.ArrayLike = np.array(np.matmul(matW, self.matBasisV))

        # shift the center, c' = c' + b
        matBasisNewV[:, 0] = matBasisNewV[:, 0] + cvecb

        # Create affine map of X
        objStarSet: StarSet = StarSet(matBasisNewV, self.matConstraintC, self.cvecConstraintd)

        # Return affine StarSet
        return objStarSet

    def minkowskiSum(self, objStarSet: 'StarSet') -> 'StarSet':
        """
        Minkowski sum of two star sets
        :param objStarSet: second star set
        :type objStarSet: StarSet
        :return: (objSumStarSet -> StarSet) minkowski sum of two star sets
        """
        # Horizontal concatenation of two matrices
        matBasisSumV: npt.ArrayLike = np.concatenate((self.matBasisV, objStarSet.matBasisV[:, 1:]), axis=1)

        # Sum of centers
        matBasisSumV[:, 0] = self.matBasisV[:, 0] + objStarSet.matBasisV[:, 0]

        # Express matConstraintC of both self and objStarSet as a block diagonal matrix
        r1: int = np.shape(self.matConstraintC)[0]
        c1: int = np.shape(self.matConstraintC)[1]
        r2: int = np.shape(objStarSet.matConstraintC)[0]
        c2: int = np.shape(objStarSet.matConstraintC)[1]
        matConstraintC: npt.ArrayLike = np.block([[self.matConstraintC, np.zeros((r1, c2))],
                                                  [np.zeros((r2, c1)), objStarSet.matConstraintC]])

        # Vertical concatenation of column vectors cvecConstraintd of both self and objStarSet
        cvecConstraintd: npt.ArrayLike = np.concatenate((self.cvecConstraintd, objStarSet.cvecConstraintd), axis=0)

        # Create an instance of StarSet for the minkowski sum
        objSumStarSet: StarSet = StarSet(matBasisSumV, matConstraintC, cvecConstraintd)

        # Return minkowski sum of two star sets
        return objSumStarSet

    def intersectPositiveHalfSpaceByIndex(self, intIndex: int) -> 'StarSet':
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]>=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objIPHSStarSet -> StarSet) intersection between self and x[intIndex]>=0
        """
        # Compute predicate matrix for the intersection
        matConstraintIPHSC: npt.ArrayLike = np.concatenate((self.matConstraintC,
                                                            np.array([-self.matBasisV[intIndex, 1:]])), axis=0)

        # Compute column vector for the predicate constraint
        cvecConstraintIPHSd: npt.ArrayLike = np.concatenate((self.cvecConstraintd,
                                                             np.array([[self.matBasisV[intIndex][0]]])), axis=0)

        # Create star set for the intersection
        objIPHSStarSet: StarSet = StarSet(self.matBasisV, matConstraintIPHSC, cvecConstraintIPHSd)

        # Return the star set for the intersection
        return objIPHSStarSet

    def intersectNegativeHalfSpaceByIndex(self, intIndex: int) -> 'StarSet':
        """
        Compute intersection between self and a specific half space by index
        intersection(self x[index]<=0)
        :param intIndex: index of the state variables
        :type intIndex: int
        :return: (objIPHSStarSet -> StarSet) intersection between self and x[intIndex]>=0
        """
        # Compute predicate matrix for the intersection
        matConstraintINHSC: npt.ArrayLike = np.concatenate((self.matConstraintC,
                                                            np.array([self.matBasisV[intIndex, 1:]])), axis=0)

        # Compute column vector for the predicate constraint
        cvecConstraintINHSd: npt.ArrayLike = np.concatenate((self.cvecConstraintd,
                                                             np.array([[-self.matBasisV[intIndex][0]]])), axis=0)

        # Create star set for the intersection
        objINHSStarSet: StarSet = StarSet(self.matBasisV, matConstraintINHSC, cvecConstraintINHSd)

        # Return the star set for the intersection
        return objINHSStarSet

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



