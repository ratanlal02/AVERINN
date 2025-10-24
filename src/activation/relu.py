"""
Author: Ratan Lal
Date : January 28, 2025
"""
from typing import List

from src.set.intervalstarset import IntervalStarSet
from src.set.set import Set
from src.types.datatype import DataType


class Relu:
    """
    perform the computation of different activation functions
    """

    @staticmethod
    def point(floatValue: DataType.RealType) -> DataType.RealType:
        """
        compute relu of a real value
        :param floatValue: a real value
        :type floatValue: float
        :return: (floatRelu -> float)
        """
        floatRelu: DataType.RealType = DataType.RealType(0.0)
        if floatValue >= 0.0:
            floatRelu = floatValue

        # return relu of floatValue
        return floatRelu

    @staticmethod
    def anySet(objSet: Set) -> List[Set]:
        """
        Compute Relu operation of any set
        :param objSet: an instance of the Set class
        :type objSet: Set
        :return: (listSets -> List[Set])
        """
        intDim: int = objSet.getDimension()
        listSets: List[Set] = [objSet]
        for i in range(intDim):
            tempListSets: List[Set] = []
            for S in listSets:
                objSetIPHSi: Set = S.intersectPHSByIndex(i)
                if not(objSetIPHSi.isEmpty()):
                    tempListSets.append(objSetIPHSi)
                objSetINHSi: Set = S.intersectNHSByIndex(i)
                if not(objSetINHSi.isEmpty()):
                    tempListSets.append(objSetINHSi)
            listSets = tempListSets

        return listSets



