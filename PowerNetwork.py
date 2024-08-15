import pandas as pd
import numpy as np

#Created by Janardan Rimal

class Network:
    def __init__(self, path=""):
           print("This is the path: "+path)
           self.workbook = path
           self.buses = None         
           self.lines = None         
           self.generators = None           
           self.tapPositions = None
           self.shuntCapacitors = None
           self.busCount = 0
           self.lineCount = 0
           self.generatorCount = 0
           self.transformerCount = 0
           self.shuntcapacitorCount = 0
           self.initBuses()
           self.initLines()
           self.initGenerators()
           self.initTransformers()
           self.initShuntCapacitors()
        

    def initBuses(self):
        ## populate bus parameters
        busData = pd.read_excel(self.workbook, sheet_name='BusData', header=0)
        print(busData.head(5))
        self.buses=busData.drop(0).values.tolist()
        self.busCount=len(self.buses)
        ##print(f"{self.busCount} buses configured in the network are as follows:")
        ##print(self.buses)


    def initLines(self):
        ## populate line parameters
        lineData = pd.read_excel(self.workbook, sheet_name='LineData')
        print(lineData.head())
        self.lines = lineData.drop(0).values.tolist()
        self.lineCount=len(self.lines)
        ##print(f"{self.lineCount} lines configured in the network are as follows:")
        ##print(self.lines)

    def initGenerators(self):
        ## populate generator parameters
        generatorData = pd.read_excel(self.workbook, sheet_name='GeneratorData')
        print(generatorData.head())
        self.generators=generatorData.values.tolist()
        self.generatorCount=len(self.generators)
        ##print(f"{self.generatorCount} generators configured in the network are as follows:")
        ##print(self.generators)


    def initTransformers(self):
        ## populate tap position parameters
        tapData = pd.read_excel(self.workbook, sheet_name='TapData')
        print(tapData.head())
        self.tapPositions=tapData.values.tolist()
        self.transformerCount=len(self.tapPositions)
        ##print(f"{self.transformerCount} transformer tap positions configured in the network are as follows:")
        ##print(self.tapPositions)


    def initShuntCapacitors(self):
        ## populate shunt capacitor parameters
        shuntcapacitorData = pd.read_excel(self.workbook, sheet_name='ShuntCapacitorData')
        print(shuntcapacitorData.head())
        self.shuntCapacitors=shuntcapacitorData.values.tolist()
        self.shuntcapacitorCount=len(self.shuntCapacitors)
        ##print(f"{self.shuntcapacitorCount} shunt capacitors configured in the network are as follows:")
        ##print(self.shuntCapacitors)

    def getBuses(self):
        return self.buses
    
    def getLines(self):
        return self.lines
    
    def getGenerators(self):
        return self.generators
    
    def getTappositions(self):
        return self.tapPositions
    
    def getShuntCapacitors(self):
        return self.shuntCapacitors
    
    def getBusCount(self):
        return self.busCount
    
    def getLineCount(self):
        return self.lineCount
    
    def getGeneratorCount(self):
        return self.generatorCount
    
    def getTransformerCount(self):
        return self.transformerCount
    
    def getCapacitorCount(self):
        return self.shuntcapacitorCount


