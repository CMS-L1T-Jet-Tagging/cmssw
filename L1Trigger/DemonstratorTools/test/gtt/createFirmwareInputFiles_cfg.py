import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing


# PART 1 : PARSE ARGUMENTS

options = VarParsing.VarParsing ('analysis')
options.register ('format',
                  'EMP', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "File format (APx, EMP or X20)")
options.parseArguments()

inputFiles = []
for filePath in options.inputFiles:
    if filePath.endswith(".root"):
        inputFiles.append(filePath)
    else:
        inputFiles += FileUtils.loadListFromFile(filePath)


# PART 2: SETUP MAIN CMSSW PROCESS 

process = cms.Process("GTTInputWriter")

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(inputFiles) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")
process.load('L1Trigger.L1TTrackMatch.L1GTTInputProducer_cfi')
#process.load("L1Trigger.L1TTrackMatch.L1TrackJetProducer_cfi")
#process.load("L1Trigger.L1TTrackMatch.L1TrackFastJetProducer_cfi")
#process.load("L1Trigger.L1TTrackMatch.L1TrackerEtMissProducer_cfi")
#process.load("L1Trigger.L1TTrackMatch.L1TkHTMissProducer_cfi")
process.load('L1Trigger.VertexFinder.VertexProducer_cff')
process.load('L1Trigger.DemonstratorTools.GTTInputFileWriter_cff')

process.VertexProducer.l1TracksInputTag = cms.InputTag("L1GTTInputProducer","Level1TTTracksConverted")
process.VertexProducer.VertexReconstruction.Algorithm = cms.string("FastHistoEmulation")

process.GTTInputFileWriter.format = cms.untracked.string(options.format)
# process.GTTInputFileWriter.outputFilename = cms.untracked.string("myOutputFile.txt")

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))

process.p = cms.Path(process.L1HybridTracks * process.L1GTTInputProducer * process.VertexProducer * process.GTTInputFileWriter)
