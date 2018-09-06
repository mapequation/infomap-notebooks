#%%
from collections import namedtuple, defaultdict
import re
import numpy as np
from sklearn import preprocessing
from sklearn import cluster
import time
from pathlib import Path
from itertools import islice

Link = namedtuple('Link', 'source, target, weight')

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

class StateNode(object):
    def __init__(self, stateId, physicalId, name):
        self.stateId = stateId
        self.physicalId = physicalId
        self.name = name
        self.outWeight = 0
        self.weight = 1.0
        self.stateLinks = defaultdict(float)

    def __str__(self):
        return "stateId: {}, physicalId: {}, name: {}, outWeight: {}, #stateLinks: {}".format(self.stateId, self.physicalId, self.name, self.outWeight, len(self.stateLinks))

    def addStateLink(self, stateTarget, weight):
        self.stateLinks[stateTarget] += weight
        self.outWeight += weight

    def outDegree(self):
        return len(self.stateLinks)

    def isDangling(self):
        return len(self.stateLinks) == 0


class LumpedStateNode(object):
    def __init__(self, physicalId=-1, lumpedStateId=-1, clusterId=-1):
        self.lumpedStateId = lumpedStateId
        self.physicalId = physicalId
        self.clusterId = clusterId
        self.stateIds = []
        self.outWeight = 0
        self.stateLinks = defaultdict(float)
        self.stateLinkMultiplicity = defaultdict(int)

    def __str__(self):
        return "LumpedStateNode (physicalId: {}, stateIds: {}, #links: {}".format(self.physicalId, self.stateIds, len(self.stateLinks))

    def addLumpedStateLink(self, lumpedStateTarget, weight):
        numLinksBefore = len(self.stateLinks)
        self.stateLinks[lumpedStateTarget] += weight
        self.stateLinkMultiplicity[lumpedStateTarget] += 1
        self.outWeight += weight
        numLinksAfter = len(self.stateLinks)
        return numLinksAfter - numLinksBefore

    def averageLumpedStateLinkWeights(self):
        totalOutWeight = 0.0
        for linkTarget in self.stateLinks.keys():
            weight = self.stateLinks[linkTarget]
            averageLinkWeight = weight / self.stateLinkMultiplicity[linkTarget]
            self.stateLinks[linkTarget] = averageLinkWeight
            totalOutWeight += averageLinkWeight
        self.outWeight = totalOutWeight

    def isDangling(self):
        return len(self.stateLinks) == 0


class PhysNode(object):
    def __init__(self, physicalId=-1, name=None):
        self.physicalId = physicalId
        self.name = name
        self.stateNodes = []
        self.clusters = {}  # stateId -> clusterIndex
        self.numClusters = 0
        self.lumpedStateNodes = {} # clusterIndex -> LumpedStateNode
        # self.outWeight = 0.0

    def __str__(self):
        return "Physical node {} ({} stateNodes)".format(self.stateNodes[0].physicalId if len(self.stateNodes) > 0 else '-', len(self.stateNodes))

    def numStateNodes(self):
        return len(self.stateNodes)

    def numDanglingStateNodes(self):
        return len(self.getDanglingStateNodes())

    def getDanglingStateNodes(self):
        return [n for n in self.stateNodes if n.isDangling()]

    def getName(self):
        return self.name or self.physicalId

    def addStateNode(self, node):
        self.stateNodes.append(node)

    def getLumpedNodeFromStateNodeId(self, stateId):
        clusterIndex = self.clusters[stateId]
        lumpedNode = self.lumpedStateNodes[clusterIndex]
        return lumpedNode

    def createLumpedStateNodesFromClustering(self):
        # numLumpedNodes = self.numClusters
        self.lumpedStateNodes = {}
        for stateNode in self.stateNodes:
            clusterIndex = self.clusters[stateNode.stateId]
            try:
                lumpedNode = self.lumpedStateNodes[clusterIndex]
            except KeyError:
                lumpedNode = LumpedStateNode(
            self.physicalId, clusterId=clusterIndex)
                self.lumpedStateNodes[clusterIndex] = lumpedNode
                
            lumpedNode.stateIds.append(stateNode.stateId)


class StateNetwork(object):
    def __init__(self):
        self.physNodes = defaultdict(PhysNode)
        self.stateNodes = defaultdict(StateNode)
        self.numLinks = 0
        self.totalWeight = 0.0
        self.lumpedStateNodes = defaultdict(LumpedStateNode)
        self.numClusters = 0

    def __str__(self):
        return "StateNetwork ({} physical nodes, {} state nodes and {} links)".format(len(self.physNodes), len(self.stateNodes), self.numLinks)

    def numPhysicalNodes(self):
        return len(self.physNodes)

    def numStateNodes(self):
        return len(self.stateNodes)

    def numLumpedStateNodes(self):
        return len(self.lumpedStateNodes)

    def addPhysicalNode(self, physicalId, name=None):
        physNode = self.physNodes[physicalId]
        physNode.physicalId = physicalId
        physNode.name = name
        return physNode

    def addStateNode(self, node):
        if node.stateId in self.stateNodes:
            return
        physNode = self.addPhysicalNode(node.physicalId)
        physNode.addStateNode(node)
        self.stateNodes[node.stateId] = node

    def addStateLink(self, link):
        # self.links.append(link)
        # physTarget = self.stateNodes[link.target].physicalId
        # self.stateNodes[link.source].addPhysLink(physTarget, link.weight)
        outDegreeBefore = self.stateNodes[link.source].outDegree()
        self.stateNodes[link.source].addStateLink(link.target, link.weight)
        outDegreeAfter = self.stateNodes[link.source].outDegree()

        self.numLinks += outDegreeAfter - outDegreeBefore
        self.totalWeight += link.weight
        # physSource = self.stateNodes[link.source].physicalId
        # if physSource != physTarget:
        #   self.physNodes[physSource].outWeight += link.weight

    def addLumpedStateNode(self, lumpedStateNode):
        lumpedStateId = lumpedStateNode.lumpedStateId
        if lumpedStateId == -1:
            lumpedStateId = len(self.lumpedStateNodes) + 1
            lumpedStateNode.lumpedStateId = lumpedStateId
        self.lumpedStateNodes[lumpedStateId] = lumpedStateNode

    def clearLumpedNodes(self):
        self.lumpedStateNodes.clear()
        for physNode in self.physNodes.values():
            physNode.lumpedStateNodes = {}
    # def getLumpedNodeFromStateNodeId(self, stateId):
    #   physId = self.stateNodes[stateId].physicalId
    #   return self.physNodes[physId].getLumpedNodeFromStateNodeId(stateId)

    def generateStateNetworkFromPaths(self, inputFilename, outputFilename, outputValidationFilename=None, markovOrder=2, validationProb=0.5, splitWeight=True, minPathLength=None, maxPathLength=None):
        """Read path data and generate second order state network

        @param inputFilename : string, path to file with *paths data
        @param outputFilename : string, path to output state network
        @param outputValidationFilename : string, path to validation state network. If not None, the paths would be split into a training and a validation state network, keeping same stateId for state nodes with same physical n-gram, and non-overlapping state ids for state nodes unique to one set.
        @param markovOrder : int, markov order of generated state network (default: 2) 
        @param validationProb : float, probability to save a path to the validation network
        @param splitWeight : bool, treat a path with weight n as n paths of weight 1 and save each individual path to validation network with probability validationProb
        """
        context = None
        print("Read path data from file '{}'...".format(inputFilename))
        np.random.seed(2)
        ngramToStateId = {}
        stateNetwork = StateNetwork()
        validationNetwork = StateNetwork()
        createValidationNetwork = outputValidationFilename is not None
        with open(inputFilename, 'r') as fp:
            for line in fp:
                if line.startswith('#'):
                    continue
                if line.startswith('*'):
                    l = line.lower()
                    if l.startswith('*paths'):
                        context = 'Paths'
                        continue
                    elif l.startswith('*vertices'):
                        context = 'Vertices'
                        continue
                    else:
                        context = '-'
                        continue
                # if context == 'Vertices':
                #     m = re.match(r'(\d+)(?: \"(.+)\")?', line)
                #     if m:
                #         [physicalId, name] = m.groups()
                #         node = PhysNode(int(physicalId), name)
                #         self.addPhysicalNode(node)
                if context == 'Paths':
                    pathStr = line.split()
                    weight = int(pathStr.pop())
                    length = len(pathStr)
                    pathNotOk = length <= markovOrder or (maxPathLength and length > maxPathLength) or (minPathLength and length < minPathLength)
                    if pathNotOk:
                        continue
                    path = [int(p) for p in pathStr]
                    weightValidation = 0
                    if createValidationNetwork:
                        if splitWeight:
                            weightValidation = np.random.binomial(weight, validationProb)
                        else:
                            weightValidation = weight if np.random.random() < validationProb else 0
                    weightTraining = weight - weightValidation
                    addValidation = weightValidation > 0
                    addTraining = weightTraining > 0
                    # print("path:", path)
                    prevStateId = None
                    for ngram in window(path, markovOrder):
                        # print(" -> ngram:", ngram)
                        try:
                            stateId = ngramToStateId[ngram]
                        except KeyError:
                            stateId = len(ngramToStateId) + 1
                            ngramToStateId[ngram] = stateId
                        
                        # Create state node
                        if addTraining:
                            stateNode = StateNode(stateId, ngram[-1], ' '.join(map(str,ngram)))
                            stateNetwork.addStateNode(stateNode)
                            # print("  -> Add training state node:", stateNode)
                        if addValidation:
                            stateNode = StateNode(stateId, ngram[-1], ' '.join(map(str,ngram)))
                            validationNetwork.addStateNode(stateNode)
                            # print("  -> Add validation state node:", stateNode)

                        if prevStateId is None:
                            prevStateId = stateId
                        else:
                            # Add link
                            if addTraining:
                                link = Link(prevStateId, stateId, weightTraining)
                                stateNetwork.addStateLink(link)
                                # print("   => Add training link:", link, '-> numLinks:', stateNetwork.numLinks)
                            if addValidation:
                                link = Link(prevStateId, stateId, weightValidation)
                                validationNetwork.addStateLink(link)
                                # print("   => Add validation link:", link)
                            prevStateId = stateId
        print("Generated {}state network: {}".format("training " if createValidationNetwork else "", stateNetwork))
        # print("Writing {}state network to file '{}...'".format("training " if createValidationNetwork else "", outputFilename))
        stateNetwork.writeStateNetwork(outputFilename)
        if createValidationNetwork:
            # print("Writing validation state network to file '{}...'".format(outputFilename))
            print("Generated validation state network: {}".format(validationNetwork))
            validationNetwork.writeStateNetwork(outputValidationFilename)
        # print(ngramToStateId)
        # print("Training:", stateNetwork)
        # print("Validation:", validationNetwork)
        print("Done!")
        

    def readFromFile(self, filename):
        context = None
        print("Read state network from file '{}'...".format(filename))
        with open(filename, 'r') as fp:
            for line in fp:
                if line.startswith('#'):
                    continue
                if line.startswith('*'):
                    l = line.lower()
                    if l.startswith('*states'):
                        context = 'States'
                        continue
                    elif l.startswith('*links'):
                        context = 'Links'
                        continue
                    elif l.startswith('*arcs'):
                        context = 'Links'
                        continue
                    else:
                        context = '-'
                        continue
                if context == 'States':
                    m = re.match(r'(\d+) (\d+)(?: \"(.+)\")?', line)
                    if m:
                        [stateId, physicalId, name] = m.groups()
                        # self.stateNodes.append(StateNode(int(stateId), int(physicalId), name))
                        node = StateNode(int(stateId), int(physicalId), name)
                        self.addStateNode(node)
                if context == 'Links':
                    m = re.match(r'(\d+) (\d+) ([\d\.]+)', line)
                    if m:
                        [source, target, weight] = m.groups()
                        link = Link(int(source), int(target), float(weight))
                        self.addStateLink(link)
        print(" -> {}".format(self))

    def writeStateNetwork(self, filename):
        print("Writing state network to file '{}'...".format(filename))
        with open(filename, 'w') as fp:
            fp.write("# physical nodes: {}\n".format(self.numPhysicalNodes()))
            fp.write("# state nodes: {}\n".format(self.numStateNodes()))
            # vertices
            fp.write("*Vertices\n")
            for physId, physNode in self.physNodes.items():
                fp.write("{} \"{}\"\n".format(physId, physNode.getName()))
            # states
            fp.write("*States\n")
            fp.write("#stateId physicalId name\n")
            for stateId, stateNode in self.stateNodes.items():
                fp.write("{} {} \"{}\"\n".format(stateId,
                                                 stateNode.physicalId, stateNode.name))
            # links
            fp.write("*Links\n")
            for sourceId, stateNode in self.stateNodes.items():
                for targetId, weight in stateNode.stateLinks.items():
                    fp.write("{} {} {}\n".format(sourceId, targetId, weight))

    def writeLumpedStateNetwork(self, filename):
        print("Writing lumped state network to file '{}'...".format(filename))
        with open(filename, 'w') as fp:
            fp.write("# physical nodes: {}\n".format(self.numPhysicalNodes()))
            fp.write("# state nodes: {}\n".format(self.numStateNodes()))
            fp.write("# lumped state nodes: {}\n".format(
                self.numLumpedStateNodes()))
            # vertices
            fp.write("*Vertices\n")
            for physId, physNode in self.physNodes.items():
                fp.write("{} \"{}\"\n".format(physId, physNode.getName()))
            # states
            fp.write("*States\n")
            fp.write("#lumpedStateId physicalId lumpedStateIds\n")
            for lumpedStateId, lumpedStateNode in self.lumpedStateNodes.items():
                fp.write("{} {} \"{}\"\n".format(lumpedStateId,
                                                 lumpedStateNode.physicalId, lumpedStateNode.stateIds))
            # links
            fp.write("*Links\n")
            for sourceId, lumpedStateNode in self.lumpedStateNodes.items():
                for targetId, weight in lumpedStateNode.stateLinks.items():
                    fp.write("{} {} {}\n".format(sourceId, targetId, weight))
        

    def calcEntropyRate(self):
        h = 0.0
        for stateNode in self.stateNodes.values():
            H = 0.0
            for w in stateNode.stateLinks.values():
                p = w / stateNode.outWeight
                H -= p * np.log2(p)
            h += stateNode.outWeight * H / self.totalWeight
        return h

    def calcLumpedEntropyRate(self):
        h = 0.0
        totalWeight = 0.0
        for stateNode in self.lumpedStateNodes.values():
            totalWeight += stateNode.outWeight
        for stateNode in self.lumpedStateNodes.values():
            H = 0.0
            for w in stateNode.stateLinks.values():
                p = w / stateNode.outWeight
                H -= p * np.log2(p)
            h += stateNode.outWeight * H / totalWeight
        return h

    def getFeatureMatrix(self, physicalId, normalizeRows=True,
                         physicalFeatures=False):
        """Generate a feature matrix of outgoing link weight
        distributions per state node.
        Rows are state nodes, columns are linked target nodes

        @param physicalId : int, get feature matrix for the selected physical node
        @param normalizeRows : bool, normalize outgoing weights to a probability 
        distribution for each state node (l1-norm) (default: True)
        @param physicalFeatures : bool, aggregate outgoing links to different 
        physical nodes (reduces feature space)

        @return (X, T), where
        X is the feature matrix (np.array) of size
        (numNonDanglingStateNodes, numLinkedNodes)
        T a dictionary transforming row index to state node id
        """
        stateIdToRowIndex = defaultdict(int)
        targetIdToFeatureIndex = defaultdict(int)
        rowIndexToStateId = {}
        denseLinks = []
        physNode = self.physNodes[physicalId]
        for stateNode in physNode.stateNodes:
            # Skip dangling nodes
            if stateNode.isDangling():
                continue
            # row mapping: stateId to dense row index
            rowIndex = len(stateIdToRowIndex)
            if stateNode.stateId in stateIdToRowIndex:
                rowIndex = stateIdToRowIndex[stateNode.stateId]
            else:
                stateIdToRowIndex[stateNode.stateId] = rowIndex
            rowIndexToStateId[rowIndex] = stateNode.stateId

            for targetId, weight in stateNode.stateLinks.items():
                if physicalFeatures:
                    targetId = self.stateNodes[targetId].physicalId
                # feature mapping: physical link target to dense column index
                featureIndex = len(targetIdToFeatureIndex)
                if targetId in targetIdToFeatureIndex:
                    featureIndex = targetIdToFeatureIndex[targetId]
                else:
                    targetIdToFeatureIndex[targetId] = featureIndex
                denseLinks.append((rowIndex, featureIndex, weight))
        numRows, numFeatures = (len(stateIdToRowIndex),
                                len(targetIdToFeatureIndex))
        X = np.zeros((numRows, numFeatures))
        if numFeatures is 0:
            return X, {}
        for (rowIndex, featureIndex, weight) in denseLinks:
            X[rowIndex][featureIndex] += weight

        if normalizeRows:
            preprocessing.normalize(X, axis=1, norm='l1', copy=False)

        return X, rowIndexToStateId

    def clusterStateNodes(self, physicalNodeIds=None,
                          physicalFeatures=False, clusterFeatureMatrix=None,
                          clusterRate=0.5, getNumClusters=None,
                          mergeDanglingNodes=True,
                          skipLumping=False):
        """Generate a cluster map for all state nodes that is used when lumping them
        
        @param physicalNodeIds : list, cluster only selected physical nodes
        @param physicalFeatures : bool, aggregate outgoing links to different 
        physical nodes (reduces feature space, default: False)
        @param clusterFeatureMatrix : callable, function that takes the feature matrix X as input and should return the clustering labels as a list
        @param clusterRate : float, if no clusterFeatureMatrix or getNumClusters is provided, use default clustering method with the number of clusters set to clusterRate times the number of state nodes.
        @param getNumClusters : callable, function that takes numStates as input and should return the number of clusters
        @param mergeDanglingNodes : bool, put dangling nodes within same physical node into same cluster if true, else put them in their own clusters
        @param skipLumping : bool, don't generate lumped network from clustering after clustering is done (default: False)

        """
        print("Cluster state nodes...")
        totNumClusters = 0
        physNodeIds = physicalNodeIds or self.physNodes.keys()
        for physId in physNodeIds:
            physNode = self.physNodes[physId]
            X, rowIndexToStateId = self.getFeatureMatrix(
                physId, physicalFeatures=physicalFeatures)
            (numStates, numFeatures) = X.shape

            labels = list(range(numStates))
            if callable(clusterFeatureMatrix):
                labels = clusterFeatureMatrix(X)
            else:
                if numStates < 2 or numFeatures < 2:
                    labels = list(range(numStates))
                else:
                    n_clusters = getNumClusters(numStates) if callable(
                        getNumClusters) else max(1, int(clusterRate * numStates))
                    model = cluster.AgglomerativeClustering(
                        linkage="complete",
                        affinity="cosine",
                        n_clusters=n_clusters
                    )
                    labels = model.fit_predict(X)

            clusters = {}
            maxClusterIndex = -1
            for rowIndex, clusterIndex in enumerate(labels):
                clusters[rowIndexToStateId[rowIndex]] = clusterIndex
                if clusterIndex > maxClusterIndex:
                    maxClusterIndex = clusterIndex
            danglingStateNodes = physNode.getDanglingStateNodes()
            numDanglingStateNodes = len(danglingStateNodes)
            if numDanglingStateNodes > 0:
                if mergeDanglingNodes:
                    # add dangling nodes to a separate last cluster
                    maxClusterIndex += 1
                    danglingClusterIndex = maxClusterIndex
                    for danglingStateNode in danglingStateNodes:
                        clusters[danglingStateNode.stateId] = danglingClusterIndex
                else:
                    # add dangling nodes to their own cluster
                    clusterIndex = maxClusterIndex + 1
                    maxClusterIndex += numDanglingStateNodes
                    for danglingStateNode in danglingStateNodes:
                        clusters[danglingStateNode.stateId] = clusterIndex
                        clusterIndex += 1

            physNode.clusters = clusters
            numClusters = maxClusterIndex + 1
            physNode.numClusters = numClusters
            totNumClusters += numClusters

        self.numClusters = totNumClusters
        if skipLumping:
            print("Done!")
        else:
            self.generateLumpedNetwork()

    def clusterStateNodesFromNetwork(self, network, skipLumping=False):
        """Cluster state nodes from clustering in another network
        @param network : StateNetwork, use clustering from network, mapping same input state ids to same lumped state ids. State nodes not in input network will not get lumped unless dangling nodes and will get state ids not among the lumped state ids.
        @param skipLumping : bool, don't generate lumped network from clustering after clustering is done (default: False)
        """
        if network.numClusters == 0:
            raise RuntimeError(
                "No clusters in input network, did you forgot to run clustering before?")
        print("Cluster state nodes from clustering in network {}...".format(network))
        
        self.clearLumpedNodes()
        
        uniqueLumpedId = network.numLumpedStateNodes() + 1
        physIdToClusterIdToLumpedStateId = {}
        totNumClusters = 0
        for physId, physNode in self.physNodes.items():
            uniqueDanglingNodes = []
            uniqueNonDanglingNodes = []
            clusters = {}
            clusterIds = set()
            uniqueClusterId = 0
            clusterIdToLumpedStateId = {}
            if not physId in network.physNodes:
                # Physical node doesn't exist in other network, add all state nodes to list of unique
                # print("\nphysId {} unique!".format(physId))
                for stateNode in physNode.stateNodes:
                    if stateNode.isDangling():
                        uniqueDanglingNodes.append(stateNode)
                    else:
                        uniqueNonDanglingNodes.append(stateNode)
            else:
                # Physical node exist in other network, map same state nodes to same cluster
                physNode2 = network.physNodes[physId]
                clusters2 = physNode2.clusters
                numClusters2 = physNode2.numClusters
                uniqueClusterId = numClusters2

                # print("\nphysId: {}, clusters2: {}".format(physId, clusters2))

                for stateNode in physNode.stateNodes:
                    try:
                        clusterId2 = clusters2[stateNode.stateId]
                        clusters[stateNode.stateId] = clusterId2
                        clusterIds.add(clusterId2)
                        # Use same lumped state id as in other network
                        clusterIdToLumpedStateId[clusterId2] = physNode2.lumpedStateNodes[clusterId2].lumpedStateId
                    except KeyError:
                        if stateNode.isDangling():
                            uniqueDanglingNodes.append(stateNode)
                        else:
                            uniqueNonDanglingNodes.append(stateNode)

            # Put unique state nodes in their own lumped node
            for stateNode in uniqueNonDanglingNodes:
                clusters[stateNode.stateId] = uniqueClusterId
                clusterIds.add(uniqueClusterId)
                clusterIdToLumpedStateId[uniqueClusterId] = uniqueLumpedId
                uniqueClusterId += 1
                uniqueLumpedId += 1

            # Lump dangling nodes
            for stateNode in uniqueDanglingNodes:
                clusters[stateNode.stateId] = uniqueClusterId
                clusterIds.add(uniqueClusterId)
                clusterIdToLumpedStateId[uniqueClusterId] = uniqueLumpedId

            physNode.clusters = clusters
            physIdToClusterIdToLumpedStateId[physId] = clusterIdToLumpedStateId

            # physNode.clusters = clusters
            numClusters = len(clusterIds)
            physNode.numClusters = numClusters
            totNumClusters += numClusters

            # print(" -> {} clusters: {}, uniqueNonDanglingNodes: {}, uniqueDanglingNodes: {}".format(numClusters, clusters, [d.stateId for d in uniqueNonDanglingNodes], [d.stateId for d in uniqueDanglingNodes]))

        self.numClusters = totNumClusters
        if skipLumping:
            print("Done!")
        else:
            self.generateLumpedNetwork()

        # print("Generate lumped state network from clustering...")
        # self.clearLumpedNodes()
        # # First generate all lumped state nodes
        # for physId, physNode in self.physNodes.items():
        #     physNode.createLumpedStateNodesFromClustering()
        #     for lumpedNode in physNode.lumpedStateNodes.values():
        #         self.addLumpedStateNode(lumpedNode)
        

    def generateLumpedNetwork(self, physIdToClusterIdToLumpedStateId=None):
        """Generate lumped state network from clustering

        @param physIdToClusterIdToLumpedStateId : {{}}, set lumped state id from this if not None, otherwise generate default sequence
        """
        if self.numClusters == 0:
            raise RuntimeError(
                "No clusters, did you forgot to run clustering before?")
        print("Generate lumped state network from clustering...")
        self.clearLumpedNodes()
        # First generate all lumped state nodes
        for physId, physNode in self.physNodes.items():
            physNode.createLumpedStateNodesFromClustering()
            for lumpedNode in physNode.lumpedStateNodes.values():
                if physIdToClusterIdToLumpedStateId is not None:
                    lumpedNode.lumpedStateId = physIdToClusterIdToLumpedStateId[physId][lumpedNode.clusterId]
                self.addLumpedStateNode(lumpedNode)
        numLumpedStateLinks = 0
        # Aggregate state links to lumped state nodes
        for physId, physNode in self.physNodes.items():
            for stateNode in physNode.stateNodes:
                lumpedSourceNode = physNode.getLumpedNodeFromStateNodeId(
                    stateNode.stateId)
                for targetStateId, weight in stateNode.stateLinks.items():
                    targetPhysId = self.stateNodes[targetStateId].physicalId
                    targetPhysNode = physNode if targetPhysId == physId else self.physNodes[
                        targetPhysId]
                    lumpedTargetNode = targetPhysNode.getLumpedNodeFromStateNodeId(
                        targetStateId)
                    lumpedTargetId = lumpedTargetNode.lumpedStateId
                    numLumpedStateLinks += lumpedSourceNode.addLumpedStateLink(lumpedTargetId, weight)
        # Average instead of sum link weights
        # for stateNode in self.lumpedStateNodes.values():
        #     stateNode.averageLumpedStateLinkWeights()
        print(" -> {} state nodes and {} links in lumped network.".format(self.numLumpedStateNodes(), numLumpedStateLinks))


def calcClusters(X):
    numStates, numFeatures = X.shape
    if numStates < 2 or numFeatures < 2:
        # Don't cluster if too small
        return list(range(numStates))

    # Can be an adaptive number of clusters based on entropy reduction
    n_clusters = max(1, int(0.5 * numStates))
    model = cluster.AgglomerativeClustering(
        linkage="complete",
        # affinity=jensen_shannon_distances,
        affinity="cosine",
        n_clusters=n_clusters
    )

    labels = model.fit_predict(X)
    return labels

def testValidate():
    start = time.clock()
    
    sparseNet = StateNetwork()
    sparseNet.readFromFile("data/toy_states.net")
    
    sparseNet.clusterStateNodes(clusterFeatureMatrix=calcClusters)

    h1 = sparseNet.calcEntropyRate()
    h2 = sparseNet.calcLumpedEntropyRate()
    print("Entropy rate original: {}, lumped: {}".format(h1, h2))
    
    sparseNet.writeLumpedStateNetwork("output/toy_lumped.net")

    validationNet = StateNetwork()
    validationNet.readFromFile("data/toy_states2.net")

    validationNet.clusterStateNodesFromNetwork(sparseNet)
    h1 = validationNet.calcEntropyRate()
    h2 = validationNet.calcLumpedEntropyRate()
    print("Entropy rate original: {}, lumped: {}".format(h1, h2))
    
    validationNet.writeLumpedStateNetwork("output/toy2_lumped.net")

    
    print("Elapsed time: {}s".format(time.clock() - start))

def testPaths():
    net = StateNetwork()
    net.generateStateNetworkFromPaths("data/toy_paths.net", "output/toy_paths_states_training.net", "output/toy_paths_states_validation.net" and None, splitWeight=True, markovOrder=2)

if __name__ == '__main__':
    testPaths()

