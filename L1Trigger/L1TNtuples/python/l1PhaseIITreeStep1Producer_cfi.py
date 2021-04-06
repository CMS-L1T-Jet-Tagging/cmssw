import FWCore.ParameterSet.Config as cms

l1PhaseIITree = cms.EDAnalyzer("L1PhaseIITreeStep1Producer",

   egTokenBarrel = cms.InputTag("L1EGammaClusterEmuProducer",""),
   tkEGTokenBarrel = cms.InputTag("L1TkElectronsEllipticMatchCrystal","EG"),
   tkEMTokenBarrel = cms.InputTag("L1TkPhotonsCrystal","EG"),

   egTokenHGC = cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts"),
   tkEGTokenHGC = cms.InputTag("L1TkElectronsEllipticMatchHGC","EG"),
   tkEMTokenHGC = cms.InputTag("L1TkPhotonsHGC","EG"),

   muonKalman = cms.InputTag("simKBmtfDigis","BMTF"),
   muonOverlap = cms.InputTag("simOmtfDigis","OMTF"),
   muonEndcap = cms.InputTag("simEmtfDigis",""),
   TkMuonToken = cms.InputTag("L1TkMuons",""),

   #Global muons
   muonToken = cms.untracked.InputTag("simGmtStage2Digis", ""),
   TkGlbMuonToken = cms.InputTag("L1TkGlbMuons",""),

   scPFL1Puppi = cms.InputTag("scPFL1Puppi", ""),

   l1pfPhase1L1TJetToken  = cms.InputTag("Phase1L1TJetCalibrator9x9" ,   "Phase1L1TJetFromPfCandidates"), #use the 9x9 case
   l1pfPhase1L1TJetMET  = cms.InputTag("Phase1L1TJetProducer9x9" ,   "UncalibratedPhase1L1TJetFromPfCandidatesMET"), #use the 9x9 case
   l1pfPhase1L1TJetSums  = cms.InputTag("Phase1L1TJetSumsProducer9x9" ,   "Sums"), #use the 9x9 case

   caloJetToken = cms.InputTag("L1CaloJet","CaloJets"),
   caloJetHTTToken = cms.InputTag("L1CaloJetHTT","CaloJetHTT"),
   caloTauToken = cms.InputTag("L1CaloJet","CaloTaus"),

   l1PFMet = cms.InputTag("l1PFMetPuppi",""),

   zoPuppi = cms.InputTag("l1pfProducerBarrel","z0"),
   l1vertextdr = cms.InputTag("VertexProducer","l1vertextdr"),
   l1vertices = cms.InputTag("VertexProducer","l1vertices"),
   l1TkPrimaryVertex= cms.InputTag("L1TkPrimaryVertex",""),

   L1NNTauToken = cms.InputTag("l1NNTauProducerPuppi","L1PFTausNN"),
   L1NNTauPFToken = cms.InputTag("l1NNTauProducer","L1PFTausNN"),

   tkTrackerJetToken = cms.InputTag("L1TrackJets", "L1TrackJets"),
   tkTrackerJetDisplacedToken = cms.InputTag("L1TrackJetsExtended", "L1TrackJetsExtended"),

   tkMetToken = cms.InputTag("L1TrackerEtMiss","L1TrackerEtMiss"),
   tkMetDisplacedToken = cms.InputTag("L1TrackerEtMissExtended","L1TrackerEtMissExtended"),
   
   tkMhtTokens = cms.VInputTag( cms.InputTag("L1TrackerHTMiss","L1TrackerHTMiss")),
   tkMhtDisplacedTokens = cms.VInputTag( cms.InputTag("L1TrackerHTMissExtended","L1TrackerHTMissExtended")),

   maxL1Extra = cms.uint32(50)
)

#### Gen level tree

from L1Trigger.L1TNtuples.l1GeneratorTree_cfi  import l1GeneratorTree
genTree=l1GeneratorTree.clone()

runmenutree=cms.Path(l1PhaseIITree*genTree)




