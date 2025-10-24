"""
Author: Ratan Lal
Date : November 7, 2024
"""
from typing import Dict

from src.gnn.ireal import IReal
from src.gnn.number import Number
from src.gnn.connection import Connection
from src.gnn.edge import Edge
from src.gnn.layer import Layer
from src.gnn.gnn import GNN
from src.gnn.node import Node
from src.types.abstype import AbsType
from src.types.networktype import NetworkType
from src.parser.parseruts import ParserUTS


class Abstraction:
    """
    It captures different abstraction of GNN class
    """

    def __init__(self, objGNN: GNN, dictPartition: Dict[int, Dict[int, set[int]]],
                 absType: AbsType):
        """
        It creates an instance of GNN that will have an abstraction of another
        instance of GNN class
        :param objGNN: An instance of the GNN class
        :type objGNN: GNN
        :param dictPartition: dictionary between abstract neurons' ids and sets of concrete
        neurons' ids
        :type dictPartition: Dict[int, Dict[int, set[int]]]
        :param absType: a specific type of abstraction
        :type absType: AbsType
        """
        self.__objGNN__ = objGNN
        self.__dictPartition__ = dictPartition
        self.__absType__ = absType

    def getAbstraction(self) -> GNN:
        """
        Return the abstraction of the GNN class
        :return: (objGNNAbs -> GNN)
        Abstraction of GNN class
        """
        return self.__abstraction__()

    def getLayerWiseAbstraction(self, intLayerNum: int) -> GNN:
        """
        Return the abstraction of the GNN class between intLayerNum
        and intLayerNum + 1
        :param intLayerNum: layer number
        :type intLayerNum: int
        :return: (objGNNAbs -> GNN)
        Abstraction of GNN class
        """
        return self.__layerWiseAbstraction__(intLayerNum)

    def __abstraction__(self) -> GNN:
        """
        Find an abstraction of the GNN based on abstraction types
        :return: (objGNNAbs -> GNN)
        An abstraction of GNN instance
        """
        objGNNAbs: GNN = None
        if self.__absType__ == AbsType.INTERVAL:
            objGNNAbs: GNN = self.__intervalAbstraction__()
        # Return an abstraction of GNN
        return objGNNAbs

    def __layerWiseAbstraction__(self, intLayerNum: int) -> GNN:
        """
        Return an abstract Neural Network as an instance of GNN for
        between intLayerNum and intLayerNum+1 layers
        :param intLayerNum: layer number
        :type intLayerNum: int
        :return: (objGNNAbs -> GNN)
        """
        # Create dictionary of Layers for intLayerNum and intLayerNum + 1
        dictLayersAbs: Dict[int, Layer] = {}
        k: int = 1
        for i in range(intLayerNum, intLayerNum+2, 1):
            dictLayersAbs[k] = self.__intervalLayerByLayer__(i)
            k = k + 1
        # Create dictionary of Connection between intLayerNum and intLayerNum+1
        k = 1
        dictConnectionsAbs: Dict[int, Connection] = {}
        dictConnectionsAbs[k] = self.__intervalConnectionByLayer__(dictLayersAbs, intLayerNum)
        # Crete dictionary between Layer number and its number of neurons

        dictNumNeuronsAbs: Dict[int, int] = {}
        for i in range(1, 3, 1):
            dictNumNeuronsAbs[i] = dictLayersAbs[i].intNumNodes

        # Create NeuralNetwork instance for the abstract system
        objGNNAbs: GNN = GNN(dictLayersAbs, dictConnectionsAbs,2, dictNumNeuronsAbs,
                             NetworkType.INTERVAL)
        # Return abstract neural network between intLayerNum and intLayerNum + 1
        return objGNNAbs

    def __intervalAbstraction__(self) -> GNN:
        """
        Construct an instance of GNN class
        based on a given partition of NodeIds of another instance of GNN
        :return: (objGNNAbs -> GNN)
        An abstraction of GNN instance
        """
        # Create the following parameters for creating an Interval abstraction
        # Create dictionary between Layer number and its Layer class instance
        # Number of layers including input and output layers
        intNumLayersAbs: int = self.__objGNN__.getNumOfLayers()
        dictLayersAbs: Dict[int, Layer] = self.__intervalDictLayers__()
        # Create dictionary between Layer number and its Connection class instance
        dictConnectionsAbs: Dict[int, Connection] = self.__intervalDictConnections__(dictLayersAbs)
        # Crete dictionary between Layer number and its number of neurons
        dictNumNeuronsAbs: Dict[int, int] = self.__intervalDictNumOfNeurons__(dictLayersAbs)
        # Create NeuralNetwork instance for the abstract system
        objGNNAbs: GNN = GNN(dictLayersAbs, dictConnectionsAbs,
                             intNumLayersAbs, dictNumNeuronsAbs, NetworkType.INTERVAL)
        # Return an instance of GNN for the abstract system
        return objGNNAbs

    def __intervalDictLayers__(self) -> Dict[int, Layer]:
        """
        Construct Layer instances for interval abstraction as an instance of GNN
        dictionary between layer numbers and Layer instances for an abstract GNN
        """
        # Create a dictionary between the layer numbers and Layer instances
        # for an abstract GNN
        dictLayersAbs: Dict[int, Layer] = {}
        # Get  the dictionary between  the layer numbers and Layer instances
        # for the original GNN
        dictLayers: Dict[int, Layer] = self.__objGNN__.getDictLayers()
        # Get the number of layers from original GNN
        intNumLayers: int = self.__objGNN__.getNumOfLayers()
        for i in range(1, self.__objGNN__.getNumOfLayers() + 1, 1):
            # Dictionary of abstract Node instances for the layer i
            dictNodesAbs: Dict[int, Node] = {}
            for intIdAbs in self.__dictPartition__[i].keys():
                # Compute intervalBiasAbs
                low: float = \
                    min([dictLayers[i].dictNodes[j].bias.getLower()
                         for j in self.__dictPartition__[i][intIdAbs]])
                high: float = max([dictLayers[i].dictNodes[j].bias.getUpper()
                                   for j in self.__dictPartition__[i][intIdAbs]])
                biasAbs: Number = IReal(low, high)
                # Compute action
                j = list(self.__dictPartition__[i][intIdAbs])[0]
                enumAction = dictLayers[i].dictNodes[j].enumAction

                # Compute size for an abstract INode instance
                intSizeAbs = 0
                for j in self.__dictPartition__[i][intIdAbs]:
                    intSizeAbs += dictLayers[i].dictNodes[j].intSize

                # Create an abstract node
                dictNodesAbs[intIdAbs] = Node(enumAction, biasAbs, intSizeAbs, intIdAbs)

            # Update the dictionary for the layer
            dictLayersAbs[i] = Layer(dictNodesAbs)

        # Return  dictILayersAbs
        return dictLayersAbs

    def __intervalDictConnections__(self, dictLayersAbs: Dict[int, Layer]) \
            -> Dict[int, Connection]:
        """
        Construct Connection instances for interval abstraction as an instance of GNN
        :param dictLayersAbs:  dictionary between layer numbers and Layer instances for
        the abstract NeuralNetwork
        :type dictLayersAbs: Dict[int, Layer]
        :return: (dictConnectionsAbs -> Dict[int, Connection])
        dictionary between layer numbers and Connection instances for an abstract GNN
        """
        # Create a dictionary between layer numbers and Connection instances
        # for an abstract GNN
        dictConnectionsAbs: Dict[int, Connection] = {}
        # Get the dictionary between layer numbers and Connection instances
        # for the original GNN
        dictConnections: Dict[int, Connection] = self.__objGNN__.getDictConnections()

        # Iterate over all the layers
        intNumLayers: int = self.__objGNN__.getNumOfLayers()

        for i in range(1, intNumLayers, 1):
            # Dictionary for the abstract edges between layer i and i+1
            dictEdgesAbs: Dict[(int, int), Edge] = {}
            for intIdSourceAbs in self.__dictPartition__[i].keys():
                for intTargetIdAbs in self.__dictPartition__[i + 1].keys():
                    # Compute interval weight for an abstract edge
                    low: float = min([dictConnections[i].dictEdges[(j, k)].weight.getLower()
                                      for j in self.__dictPartition__[i][intIdSourceAbs]
                                      for k in self.__dictPartition__[i + 1][intTargetIdAbs]])
                    high: float = max([dictConnections[i].dictEdges[(j, k)].weight.getUpper()
                                       for j in self.__dictPartition__[i][intIdSourceAbs]
                                       for k in self.__dictPartition__[i + 1][intTargetIdAbs]])

                    # Create an abstract edge
                    weightAbs: Number = IReal(low, high)
                    dictEdgesAbs[(intIdSourceAbs, intTargetIdAbs)] = \
                        Edge(dictLayersAbs[i].dictNodes[intIdSourceAbs],
                             dictLayersAbs[i + 1].dictNodes[intTargetIdAbs], weightAbs)

            # update the dictIConnectionsAbs for the abstract GNN
            dictConnectionsAbs[i] = Connection(dictEdgesAbs)

        # Return dictIConnectionsAbs
        return dictConnectionsAbs

    def __intervalDictNumOfNeurons__(self, dictLayersAbs: Dict[int, Layer]) -> Dict[int, int]:
        """
        Construct dictionary between layer numbers and numbers of neurons
        for the abstract NeuralNetwork
        :param dictLayersAbs: Dictionary between layer numbers and Layer instances
        :type dictLayersAbs: Dict[int, Layer]
        :return: (dictNumOfNeurons -> Dict[int, int])
        dictionary between layer numbers and numbers of neurons for
        the abstract GNN
        """
        # Create dictionary between layer numbers and number of neurons
        dictNumOfNeuronsAbs: Dict[int, int] = {}

        # Update dictNumOfNeurons
        for intLayerNum in dictLayersAbs.keys():
            dictNumOfNeuronsAbs[intLayerNum] = dictLayersAbs[intLayerNum].intNumNodes

        # Return dictNumOfNeuronsAbs
        return dictNumOfNeuronsAbs

    def __intervalLayerByLayer__(self, intLayerNum: int) -> Layer:
        """
        Construct a Layer instance for an abstract GNN for a given layer number
        :param intLayerNum: Layer number
        :type intLayerNum: int
        :return: (objLayerAbs -> Layer)
        """
        # Get a layer instance from original GNN instance for intLayerNum
        dictLayers: Dict[int, Layer] = self.__objGNN__.getDictLayers()
        objLayer: Layer = dictLayers[intLayerNum]
        i: int = intLayerNum
        # Dictionary of abstract Node instances for the layer i
        dictNodesAbs: Dict[int, Node] = {}
        for intIdAbs in self.__dictPartition__[i].keys():
            # Compute intervalBiasAbs
            low: float = min([dictLayers[i].dictNodes[j].bias.getLower()
                              for j in self.__dictPartition__[i][intIdAbs]])
            high: float = max([dictLayers[i].dictNodes[j].bias.getUpper()
                               for j in self.__dictPartition__[i][intIdAbs]])

            biasAbs: Number = IReal(low, high)
            # Compute action
            j = list(self.__dictPartition__[i][intIdAbs])[0]
            enumAction = objLayer.dictNodes[j].enumAction
            # Compute size for an abstract Node instance
            intSizeAbs = 0
            for j in self.__dictPartition__[i][intIdAbs]:
                intSizeAbs += dictLayers[i].dictNodes[j].intSize
            # Create an abstract node
            dictNodesAbs[intIdAbs] = Node(enumAction, biasAbs, intSizeAbs, intIdAbs)

        # Create an abstract layer as a Layer instance for intLayerNum
        objLayerAbs = Layer(dictNodesAbs)
        # Return  objLayerAbs
        return objLayerAbs

    def __intervalConnectionByLayer__(self, dictLayersAbs: Dict[int, Layer], intLayerNum: int) -> Connection:
        """
        Construct an abstract Connection instance between intLayerNum and intLayerNum + 1
        :param dictLayersAbs:  dictionary between layer numbers and Layer instances for
        the abstract NeuralNetwork
        :type dictLayersAbs: Dict[int, Layer]
        :param intLayerNum: Layer number
        :type intLayerNum: int
        :return: (objConnection -> Connection)
        an abstract Connection instance between intLayerNum and intLayerNum + 1
        """
        # Get the dictionary between layer numbers and Connection instances
        # for the original GNN
        dictConnections: Dict[int, Connection] = self.__objGNN__.getDictConnections()
        objConnection: Connection = dictConnections[intLayerNum]

        # Iterate over all the layers
        i: int = intLayerNum
        # Dictionary for the abstract edges between layer i and i+1
        dictEdgesAbs: Dict[(int, int), Edge] = {}
        for intIdSourceAbs in self.__dictPartition__[i].keys():
            for intTargetIdAbs in self.__dictPartition__[i + 1].keys():
                # Compute interval weight for an abstract edge
                low: float = min([objConnection.dictEdges[(j, k)].weight.getLower()
                                  for j in self.__dictPartition__[i][intIdSourceAbs]
                                  for k in self.__dictPartition__[i + 1][intTargetIdAbs]])
                high: float = max([objConnection.dictEdges[(j, k)].weight.getUpper()
                                   for j in self.__dictPartition__[i][intIdSourceAbs]
                                   for k in self.__dictPartition__[i + 1][intTargetIdAbs]])

                # Create an abstract edge
                weightAbs: Number = IReal(low, high)
                dictEdgesAbs[(intIdSourceAbs, intTargetIdAbs)] = \
                    Edge(dictLayersAbs[1].dictNodes[intIdSourceAbs],
                         dictLayersAbs[2].dictNodes[intTargetIdAbs], weightAbs)

        # Create an abstract Connection instance between intLayerNum and intLayerNum + 1
        objConnectionAbs = Connection(dictEdgesAbs)

        # Return objConnectionsAbs
        return objConnectionAbs
