from typing import Dict, List
import numpy as np

from src.abstraction.abstraction import Abstraction
from src.evaluation.acasXUMILP import partitionType, absType
from src.gnn.gnn import GNN
from src.partition.partition import Partition
from src.rasc.iss import matlab_interface
from src.parser.nnet import Nnet
from src.parser.parser import Parser
from src.types.lastrelu import LastRelu
from src.types.techniquetype import TechniqueType
from src.types.solvertype import SolverType
from src.utilities.log import Log
from src.utilities.spec import Spec
from src.utilities.vnnlib import VNNLib
import time
from src.rasc.iss.Istarset import to_starset, reach_star_new
from src.rasc.iss.IVerification import verify_output, get_range_star


##################################
########## Parameters  ###########
##################################
solverType: SolverType = SolverType.Gurobi
techniqueType: TechniqueType = TechniqueType.MILP
lastRelu: LastRelu = LastRelu.NO

##########################################
### Dictionaries for Time and Result #####
##########################################
dictIteration: Dict[int, Dict[int, int]] = dict()
dictAbsTime: Dict[int, Dict[int, float]] = dict()
dictConfig: Dict[int, Dict[int, List[int]]] = dict()
dictVerTime: Dict[int, Dict[int, float]] = dict()
dictValidTime: Dict[int, Dict[int, float]] = dict()
dictRefTime: Dict[int, Dict[int, float]] = dict()
dictPor: Dict[int, Dict[int, int]] = dict()
dictStrategy: Dict[int, Dict[int, int]] = dict()
dictProperty: Dict[int, Dict[int, int]] = dict()

##################################
##### Initiate the log file ######
##################################
f = open("log.txt", "w")
f.close()
##################################
##### Reading ONNX and spec ######
##################################
# read acasxu benchmark files and specification
filePathAcasxu: str = \
    "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/acasxu/1.onnx"
filePath: str = \
    "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/acasxu/1.nnet"

specPath: str = \
    "/Users/ratanlal/Desktop/repositories/github/AVERINN/resources/acasxu/prop_1.vnnlib"

objParser: Parser = Nnet(filePath)

# create an instance of GNN
objGNN: GNN = objParser.getNetwork()
#objGNN.binarization()

Log.message("Original Network Structure\n")
objGNN.display()

i, o, d = VNNLib.get_num_inputs_outputs(filePathAcasxu)
ioSpec = VNNLib.read_vnnlib_simple(specPath, i, o)
objInputSet = Spec.getInput(ioSpec)
outputConstr = Spec.getOutput(ioSpec)

####################################
####  Abstraction of GNN ###########
####################################
objPartition: Partition = Partition(objGNN, partitionType, 20, None)
dictPartition: Dict[int, Dict[int, set[int]]] = objPartition.getPartition()
objAbstraction: Abstraction = Abstraction(objGNN, dictPartition, absType)
objGNNAbs: GNN = objAbstraction.getAbstraction()


####################################
########## Test Reach set ###########
####################################
stime = time.time()
inputLow = objInputSet.getLowerBound()
inputLowMatrix = np.array([[inputLow[i]] for i in range(len(inputLow))])
inputHigh = objInputSet.getUpperBound()
inputHighMatrix = np.array([[inputHigh[i]] for i in range(len(inputHigh))])
inputStarSet = to_starset(inputLowMatrix, inputHighMatrix)
stars, num_stars, times = reach_star_new(objGNNAbs, [inputStarSet], method='feasibility')
print(verify_output(stars, ioSpec[0][1]))
L, U = get_range_star(stars)
print(f'[{L[0]}, {U[0]}]')
print(f'[{L[1]}, {U[1]}]')
print(f'[{L[2]}, {U[2]}]')
print(f'[{L[3]}, {U[3]}]')
print(f'[{L[4]}, {U[4]}]')

print(num_stars)
matlab_interface.ENG.quit()
t2 = time.time()
etime: float = time.time()
print(etime - stime)
