from typing import List, Dict

from src.abstraction.abstraction import Abstraction
from src.gnn.gnn import GNN
from src.parser.onnxtonn import ONNX
from src.parser.parser import Parser
from src.partition.partition import Partition
from src.rasc.setpropagation import SetPropagation
from src.rasc.technique import Technique
from src.set.set import Set
from src.set.setuts import SetUTS
from src.types.abstype import AbsType
from src.types.lastrelu import LastRelu
from src.types.partitiontype import PartitionType
from src.types.solvertype import SolverType
from src.types.techniquetype import TechniqueType
from src.utilities.log import Log
from src.utilities.spec import Spec
from src.utilities.vnnlib import VNNLib

##################################
########## Parameters  ###########
##################################
solverType: SolverType = SolverType.Gurobi
techniqueType: TechniqueType = TechniqueType.MILP
lastRelu: LastRelu = LastRelu.NO
partitionType: PartitionType = PartitionType.FIXED
absType: AbsType = AbsType.INTERVAL

##################################
##### Initiate the log file ######
##################################
f = open("log.txt", "w")
f.close()

##################################
##### Reading ONNX and spec ######
##################################
# read acasxu benchmark files and specification
filePath: str = \
    "/Users/ratanlal/Desktop/repositories/gitlab/averinn/HSCC'24/Averinn/resources/acasxuonnx/ACASXU_run2a_1_4_batch_2000.onnx"

specPath: str = (
            "/Users/ratanlal/Desktop/repositories/gitlab/averinn/HSCC'24/Averinn/resources/acasxuvnnlibp/prop_" +
            str(1) + ".vnnlib")

objParser: Parser = ONNX(filePath)

# create an instance of GNN
objGNN: GNN = objParser.getNetwork()

Log.message("Original Network Structure\n")
objGNN.display()

i, o, d = VNNLib.get_num_inputs_outputs(filePath)
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
########### Test Reach set #########
####################################
objInputSet = SetUTS.toIntervalStarSet(objInputSet)
objTechnique: Technique = SetPropagation(objGNNAbs, objInputSet, outputConstr, solverType, lastRelu)
Log.message("Reach Set \n")
listSets: List[Set] = objTechnique.reachSet()

rangeSet: Set = SetUTS.rangeOfSets(listSets)
Log.message(str(rangeSet.getArrayLow())+"\n")
Log.message(str(rangeSet.getArrayHigh())+"\n")
Log.message("Reach set ends\n")
