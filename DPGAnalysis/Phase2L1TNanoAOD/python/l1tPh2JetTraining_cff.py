import FWCore.ParameterSet.Config as cms

# -----------------------------------------------------------------------------
# Jet tagger training tables for the Phase2 L1 NanoAOD.
#
# Reproduces, in NanoAOD flat-table format, exactly the information stored by
# FastPUPPI/NtupleProducer/plugins/JetNTuplizer.cc (jetTuple_extended_5.root).
#
# Tables produced by L1JetTrainingTableProducer (module l1JetTrainTable):
#   L1JetTrain       one row per L1 jet  (columns named as JetNTuplizer branches)
#   L1JetTrainCand   one row per puppi candidate (column jetIdx -> parent jet row)
#   L1GenH           one row per last-copy Higgs (pt, mass); nL1GenH == ngenH
#   L1GenEvent       singleton: genHH_pt, genHH_mass, genHT, genHTall
# Offline content (jet_offline_* columns and the L1JetTrainOffCand table with
# the matched slimmedJetsPuppi constituents) is DISABLED here
# (doOfflineInfo=False); set it True to recover exactly the full
# JetNTuplizer content including offline matches.
#
# Upstream configuration mirrors FastPUPPI/NtupleProducer/python/runJetNtuple.py:
#   * dedicated clone of the NGJet producer with returnRawPt=True, maxJets=999,
#     minPt=0, maxEta=99 (training/full-coverage configuration).
#     NOTE doJEC is set to False even though runJetNtuple sets True: with
#     returnRawPt=True the producer never enters the JEC code path, so the two
#     settings yield an identical output collection, while doJEC=False avoids
#     opening the JEC corrector file jecs_20220308.root which is absent from
#     the release.
#   * dedicated clone of the BJet producer running on the training NGJets
#     (same settings as runJetNtuple.addBtagging()).
#   * genFlavourInfo (JetFlavourInfoMatching for ak4GenJetsNoNu), as in
#     runJetNtuple.addGenJetFlavourTable().
# The standard menu tables (e.g. L1puppiJetSC4NG) keep using the default
# producers and are not affected.
# -----------------------------------------------------------------------------

from L1Trigger.Phase2L1ParticleFlow.l1tSC4NGJetProducer_cfi import l1tSC4NGJetProducer
from L1Trigger.Phase2L1ParticleFlow.L1BJetProducer_cfi import L1BJetProducer

# training-configured NG jets (full coverage, raw pt) --------------------------------
l1tSC4NGJetTrainProducer = l1tSC4NGJetProducer.clone(
    jets = "l1tSC4PFL1PuppiExtendedEmulator",
    returnRawPt = True,
    doJEC = False,          # see NOTE above: inactive when returnRawPt=True
    maxJets = 999,
    minPt = 0.,
    maxEta = 99.,
)

# training b-tag scores on those jets ------------------------------------------------
l1tBJetProducerTrain = L1BJetProducer.clone(
    jets = ("l1tSC4NGJetTrainProducer", "l1tSC4NGJets"),
    useRawPt = True,
    maxJets = 500,
    vtx = ("l1tVertexFinderEmulator", "L1VerticesEmulation"),
)

# gen jet flavour info for ak4GenJetsNoNu (same as runJetNtuple.addGenJetFlavourTable,
# which also sets selectedHadronsAndPartons.partonMode = "Pythia8" on the process)
from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons as _shp
selectedHadronsAndPartons = _shp.clone(partonMode = "Pythia8")
genFlavourInfo = ak4JetFlavourInfos.clone(jets = "ak4GenJetsNoNu")

# the table producer itself ----------------------------------------------------------
l1JetTrainTable = cms.EDProducer("L1JetTrainingTableProducer",
    genJets           = cms.InputTag("ak4GenJetsNoNu"),
    genParticles      = cms.InputTag("genParticles"),
    scPuppiJets       = cms.InputTag("l1tSC4NGJetTrainProducer", "l1tSC4NGJets"),
    scPuppiJetsCorr   = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulator"),
    nnTaus            = cms.InputTag("l1tNNTauProducerPuppi", "L1PFTausNN"),
    electrons         = cms.InputTag("l1tLayer2EG", "L1CtTkElectron"),
    # "prompt" is the in-job producer instance label (Phase2L1TGMTSAMuonGhost-
    # Cleaner makes "prompt"/"displaced"); JetNTuplizer-era configs read
    # "promptSAMuons" from precomputed file products (same objects).
    muons             = cms.InputTag("l1tSAMuonsGmt", "prompt"),
    genJetsFlavour    = cms.InputTag("genFlavourInfo"),
    vtx               = cms.InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"),
    bjetIDs           = cms.InputTag("l1tBJetProducerTrain", "L1PFBJets"),
    offlineJets       = cms.InputTag("slimmedJetsPuppi"),
    offlinePVs        = cms.InputTag("offlineSlimmedPrimaryVertices"),
    # OFF: keep only L1JetTrain + L1JetTrainCand (plus L1GenH/L1GenEvent).
    # With False, neither the jet_offline_* columns nor the L1JetTrainOffCand
    # table are produced (this was ~40% of the training tables' size).
    doOfflineInfo     = cms.bool(False),
)

l1tJetTrainingTask = cms.Task(
    l1tSC4NGJetTrainProducer,
    l1tBJetProducerTrain,
    selectedHadronsAndPartons,
    genFlavourInfo,
    l1JetTrainTable,
)

def addL1JetTrainingTables(process):
    """Add the jet tagger training tables to the Phase2 L1 NanoAOD production.

    Mirrors the settings of FastPUPPI/NtupleProducer/python/runJetNtuple.py:
    the NNTau cap is raised to 500 taus/event from the menu default of 5, to
    keep jet<->tau matching identical to the JetNTuplizer job.
    """
    process.l1tPh2NanoTask.add(l1tJetTrainingTask)
    if hasattr(process, "l1tNNTauProducerPuppi"):
        process.l1tNNTauProducerPuppi.maxtaus = cms.int32(500)
    return process
