from unittest import TestCase

import numpy as np
import numpy.typing as npt
from src.set.intervalmatrix import IntervalMatrix
from src.set.intervalstarset import IntervalStarSet
from src.set.set import Set


class TestIntervalStarSet(TestCase):
    def test_getMatBasisV(self):
        """
        Test getMatBasisV
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasicV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasicV, matConstraintC, arrayConstraintd)

        np.testing.assert_array_equal(IMBasicV.getMatLow(), objSet.getMatBasisV().getMatLow(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(IMBasicV.getMatHigh(), objSet.getMatBasisV().getMatHigh(),
                                      "Basic Matrices are not matched")

    def test_getMatConstraintC(self):
        """
        Test getMatConstraintC
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasicV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasicV, matConstraintC, arrayConstraintd)

        np.testing.assert_array_equal(matConstraintC, objSet.getMatConstraintC(),
                                      "Basic Matrices are not matched")

    def test_getArrayConstraintd(self):
        """
        Test getMatConstraintC
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasicV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasicV, matConstraintC, arrayConstraintd)

        np.testing.assert_array_equal(arrayConstraintd, objSet.getArrayConstraintd(),
                                      "Basic Matrices are not matched")

    def test_getNumOfPredVars(self):
        """
        Test getMatConstraintC
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasicV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasicV, matConstraintC, arrayConstraintd)

        self.assertEqual(2, objSet.getNumOfPredVars(), "Number of predicate variables are not matched")

    def test_getNumOfPredicates(self):
        """
        Test getMatConstraintC
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasicV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasicV, matConstraintC, arrayConstraintd)
        self.assertEqual(5, objSet.getNumOfPredicates(), "Number of predicates are not matched")

    def test_linearMap(self):
        """
        Test getMatConstraintC
        """
        # Create an IntervalStarSet
        matLow: npt.ArrayLike = np.array([[3, 2, 4], [2, 1, 3]], dtype=np.float64)
        matHigh: npt.ArrayLike = np.array([[4, 3, 5], [3, 2, 4]], dtype=np.float64)
        IMBasicV: IntervalMatrix = IntervalMatrix(matLow=matLow, matHigh=matHigh)
        matConstraintC: npt.ArrayLike = np.array([[1, 1], [-1, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float64)
        arrayConstraintd: npt.ArrayLike = np.array([1, 0, 0, 1, 1], dtype=np.float64)

        objSet: Set = IntervalStarSet(IMBasicV, matConstraintC, arrayConstraintd)

        # Weight Matrix
        WeightLow: npt.ArrayLike = np.array([[1, 1], [1, 1]], dtype=np.float64)
        WeightHigh: npt.ArrayLike = np.array([[1, 1], [1, 1]], dtype=np.float64)

        # Linear map
        objSetLinear: Set = objSet.linearMap(WeightLow, WeightHigh)
        print(objSetLinear.getMatBasisV().getMatLow())
        print(objSetLinear.getMatBasisV().getMatHigh())
        np.testing.assert_array_equal(IMBasicV.getMatLow(), objSetLinear.getMatBasisV().getMatLow(),
                                      "Basic Matrices are not matched")
        np.testing.assert_array_equal(IMBasicV.getMatHigh(), objSetLinear.getMatBasisV().getMatHigh(),
                                      "Basic Matrices are not matched")
