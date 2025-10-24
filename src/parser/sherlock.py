"""
Author: Ratan Lal
Date : November 4, 2023
"""
from abc import ABC
from typing import Dict, List, Tuple, IO
import numpy.typing as npt
import numpy as np

from src.gnn.gnn import GNN
from src.parser.parseruts import ParserUTS
from src.types.boundtype import BoundType
from src.parser.parser import Parser


class Sherlock(Parser, ABC):
    """
    Read neural network input expressed in a specific form taken by Sherlock tool
    """

    def __init__(self, strFilePath: str):
        # read the input file
        listContents: List[str] = self.__readFile__(strFilePath)
        # a mapping between layer number and number of neurons, current line number
        tupleConfig: Tuple[Dict[int, int], int] = self.__readConfig__(listContents)
        self.__dictNeurons__: Dict[int, int] = tupleConfig[0]
        # list of interval for input set, current line number
        # There is no input, no need to skip lines
        #  tuple of dictionary for weight, bias, current line
        tupleWeightBias: Tuple[Dict[int, npt.ArrayLike], Dict[int, npt.ArrayLike]] = \
            self.__readWeightBias__(listContents, tupleConfig[1])
        self.__dictWeights__: Dict[int, npt.ArrayLike] = tupleWeightBias[0]
        # a mapping between layer number and bias
        self.__dictBiases__: Dict[int, npt.ArrayLike] = tupleWeightBias[1]

    def getNetwork(self) -> GNN:
        """
        Get a neural network as an instance of GNN
        :return: (objGNN: GNN)
        An instance of GNN
        """

        # Return  neural network as an instance of GNN
        return ParserUTS.toStandardGNN(self.__dictNeurons__, self.__dictWeights__, self.__dictBiases__)

    def getDictNeurons(self) -> Dict[int, int]:
        """
        Get dictionary between layer number and its number of neurons
        :return: (dictNeurons: Dict[int,int])
        """
        return self.__dictNeurons__

    def getDictWeights(self, boundType: BoundType) -> Dict[int, npt.ArrayLike]:
        """
        Get dictionary between layer number i and weight matrix where each row represents
        weight array from a neuron at i+1 and all neurons at layer i
        :return: (dictWeights: Dict[int,npt.ArrayLike])
        """
        return self.__dictWeights__

    def getDictBiases(self, boundType: BoundType) -> Dict[int, npt.ArrayLike]:
        """
        Get dictionary between layer number i starting from 2 and an array of biases at layer i
        :return: (dictBiases: Dict[int,npt.ArrayLike])
        """
        return self.__dictBiases__

    def __readFile__(self, strFilePath: str) -> List[str]:
        """
        Read a neural network file and store each line
        of the file in a list of string
        :param strFilePath: path of the nn input file
        :type strFilePath: str
        :return: (contents->List[str])
         List[str] - list of strings where
         each string corresponds to a line in the file
        """
        # open file in read mode
        f: IO = open(strFilePath, "r")

        # read contents of the file
        listContents: List[str] = f.readlines()
        for intLineNumber in range(len(listContents)):
            listContents[intLineNumber] = listContents[intLineNumber].strip('\n')

        # close the file
        f.close()
        # return list of contents
        return listContents

    def __readConfig__(self, listContents: List[str]) \
            -> Tuple[Dict[int, int], int]:
        """
        Read layer and neuron information for a neural network
        :param listContents: list of strings where
        each string corresponds to a line in a neural network file
        :type listContents: List[str]
        :return: (tuple -> Tuple[Dict[int, int], int])
        Dict[int, int] - dictionary maps between layer number and its number of neurons
        int - line number from which further needs to be read to get weight/bias
        """
        # a mapping between layer number and its number of neurons
        dictNeurons: Dict[int, int] = {}
        # initialize the first layer number
        intLayerNumber: int = 1
        # initialize the first line number
        intLineNumber = 0
        # number of neurons at input layer
        dictNeurons[intLayerNumber] = int(listContents[intLineNumber])
        intLayerNumber += 1
        intLineNumber += 1
        # number of neurons at output layer
        numOfOutputNeurons: int = int(listContents[intLineNumber])
        intLineNumber += 1
        # number of hidden layers
        numOfHiddenLayers: int = int(listContents[intLineNumber])
        intLineNumber += 1
        # number of neurons at hidden layers
        for i in range(numOfHiddenLayers):
            dictNeurons[intLayerNumber] = int(listContents[intLineNumber])
            intLayerNumber += 1
            intLineNumber += 1
        # add number of output neurons
        dictNeurons[intLayerNumber] = numOfOutputNeurons

        # return dictionary and line number
        return dictNeurons, intLineNumber

    def __readWeightBias__(self, listContents: List[str], intLineNumber: int) \
            -> Tuple[Dict[int, npt.NDArray], Dict[int, npt.NDArray]]:
        """
        Read weights and biases of a neural network
        :param listContents: list of strings where
        each string corresponds to a line in a neural network file
        :type listContents: List[str]
        :param intLineNumber: line number from which further
         needs to be read to get weight/bias
        :type intLineNumber: int
        :return: (tuple -> Tuple[Dict[int, npt.ArrayLike], Dict[int, npt.ArrayLike]])
        Dict[int, npt.ArrayLike] - dictionary between layer numbers and weight matrices
        Dict[int, npt.ArrayLike] - dictionary maps between layers number and bias arrays
        """
        # a mapping between source layer number and its weight matrix
        dictWeight: Dict[int, npt.NDArray] = {}
        # a mapping between target layer number and its bias matrix
        dictBias: Dict[int, npt.NDArray] = {}
        # number of layers
        intNumLayers: int = len(self.__dictNeurons__)
        # extract weights and biases
        for i in range(1, intNumLayers, 1):
            # number of neurons at layer i
            numOfNeuronsAti: int = self.__dictNeurons__[i]
            # number of neurons at layer i+1
            numOfNeuronsAtip1: int = self.__dictNeurons__[i + 1]
            # array for the weight matrix between layer i and i+1
            arrayWeight: npt.NDArray = np.array([[0.0 for j in range(numOfNeuronsAtip1)]
                                                 for k in range(numOfNeuronsAti)])
            # extract number of neurons at layer i and layer i+1
            intNumSourceNodes = self.__dictNeurons__[i]
            intNumTargetNodes = self.__dictNeurons__[i + 1]
            # in the input file, weights are listed based on each target neuron
            arrayBias: npt.ArrayLike = np.array([0.0 for j in range(intNumTargetNodes)], dtype='f')
            for j in range(1, intNumTargetNodes + 1, 1):
                # extract weight information and create edges between nodes
                arrayWeight[:, j - 1] = arrayWeight[:, j - 1] + \
                                        np.array([[float(listContents[k])]
                                                  for k in range(intLineNumber, intLineNumber +
                                                                 intNumSourceNodes, 1)])[:, 0]
                intLineNumber += intNumSourceNodes
                # extract bias
                bias = float(listContents[intLineNumber])
                arrayBias[j - 1] = bias
                # next line
                intLineNumber += 1

            dictWeight[i] = np.transpose(arrayWeight)
            dictBias[i + 1] = arrayBias

        # return both dictionary of weight and bias
        return dictWeight, dictBias

