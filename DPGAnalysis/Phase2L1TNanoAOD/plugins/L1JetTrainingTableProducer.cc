// -*- C++ -*-
//
// Package:    DPGAnalysis/Phase2L1TNanoAOD
// Class:      L1JetTrainingTableProducer
//
/**\class L1JetTrainingTableProducer L1JetTrainingTableProducer.cc

 Description: produces nanoaod::FlatTable products carrying exactly the per-jet,
              per-candidate and event-level information stored by
              FastPUPPI/NtupleProducer/plugins/JetNTuplizer.cc ("Jets" TTree of
              jetTuple_extended_5.root), so that tagger-training information is
              available in the L1Phase2 NanoAOD format.

              Column names are kept identical to the JetNTuplizer branch names
              for a 1:1 mapping. Variable-length per-jet candidate arrays are
              stored in a flattened sibling table (rows = candidates, column
              "jetIdx" = index of the parent jet in the "L1JetTrain" table);
              the per-jet counts are in column "jet_npuppicand" /
              "njet_puppicand_off" of the jet table.

 Implementation:
     Logic is a line-by-line port of JetNTuplizer::analyze() (original author
     Sebastian Wuchterl): same selections, same thresholds, same orderings
     (std::sort by pT for jets, std::stable_sort for constituents), same
     defaults (-999 / 999 / -1 filler values).
*/

#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC4NGJetID.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/inversion.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/log.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1SeedConePFJetEmulator.h"

#include <TMath.h>
#include <TLorentzVector.h>
#include <TVector3.h>

namespace l1jettraintools {

  // sorters copied from JetNTuplizer
  template <typename T>
  class CandPtSorter {
  public:
    bool operator()(const T& i, const T& j) const { return (i.pt() > j.pt()); }
  };
  CandPtSorter<l1t::PFCandidate> l1PFCandidateSorter;

  template <typename T>
  class RefPtSorter {
  public:
    bool operator()(const T& i, const T& j) const { return (i->pt() > j->pt()); }
  };
  RefPtSorter<l1t::PFJetRef> jetRefSorter;
  RefPtSorter<l1t::PFTauRef> tauRefSorter;
  RefPtSorter<l1t::TkElectronRef> eleRefSorter;
  RefPtSorter<l1t::SAMuonRef> muonRefSorter;
  RefPtSorter<reco::GenJetRef> genJetRefSorter;
  RefPtSorter<pat::JetRef> offlineJetRefSorter;

  // Event-level GEN information, filled once per event (port of
  // JetNTuplizer::fill_genParticles + the genH/genHT block of analyze())
  struct GenEventInfo {
    bool isMC = false;
    // gen leptons / taus from resonance decays
    std::vector<TLorentzVector> genMuonsFromResonance4V;
    std::vector<TLorentzVector> genElectronsFromResonance4V;
    std::vector<TLorentzVector> tauGenVisible4V;
    std::vector<TLorentzVector> tauGen4V;
    std::vector<int> tauGenCharge;
    std::vector<unsigned int> tauGenNch, tauGenNp0, tauGenNnh;
    // Higgs / HT info
    unsigned int ngenH = 0;
    std::vector<float> genH_pt, genH_mass;
    float genHH_pt = -1, genHH_mass = -1, genHT = 0, genHTall = 0;
  };

}  // namespace l1jettraintools

class L1JetTrainingTableProducer : public edm::stream::EDProducer<> {
public:
  explicit L1JetTrainingTableProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  l1jettraintools::GenEventInfo fillGenInfo(const edm::Event& iEvent,
                                            const edm::Handle<reco::GenParticleCollection>& genParticles) const;

  // inputs, mirroring JetNTuplizer
  const edm::EDGetTokenT<std::vector<reco::GenJet>> genjetsToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genparticlesToken_;
  const edm::EDGetTokenT<std::vector<l1t::PFJet>> scjetsToken_;
  const edm::EDGetTokenT<std::vector<l1t::PFJet>> scjetsCorrToken_;
  const edm::EDGetTokenT<std::vector<l1t::PFTau>> nntausToken_;
  const edm::EDGetTokenT<std::vector<l1t::TkElectron>> electronsToken_;
  const edm::EDGetTokenT<std::vector<l1t::SAMuon>> muonsToken_;
  const edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> genJetsFlavourToken_;
  const edm::EDGetTokenT<std::vector<l1t::VertexWord>> vtxEmuToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> bjetidsToken_;
  const edm::EDGetTokenT<std::vector<pat::Jet>> offlineJetsToken_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> offlinePVsToken_;
  const bool doOfflineInfo_;

  // thresholds, identical to JetNTuplizer
  const float dRJetGenMatch_ = 0.4;
  const float jetPtMin_ = 5.;
  const float jetEtaMin_ = -999.;
  const float jetEtaMax_ = 999.;
  const float jetPFCandidatePtMin_ = 0.;
  const float dRCone_ = 0.2;
  const float ptGenLeptonMin_ = 5.;
  const float ptGenTauVisibleMin_ = 5.;
  const bool applyJetIDFlag_ = false;
};

L1JetTrainingTableProducer::L1JetTrainingTableProducer(const edm::ParameterSet& iConfig)
    : genjetsToken_(consumes<std::vector<reco::GenJet>>(iConfig.getParameter<edm::InputTag>("genJets"))),
      genparticlesToken_(consumes<std::vector<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("genParticles"))),
      scjetsToken_(consumes<std::vector<l1t::PFJet>>(iConfig.getParameter<edm::InputTag>("scPuppiJets"))),
      scjetsCorrToken_(consumes<std::vector<l1t::PFJet>>(iConfig.getParameter<edm::InputTag>("scPuppiJetsCorr"))),
      nntausToken_(consumes<std::vector<l1t::PFTau>>(iConfig.getParameter<edm::InputTag>("nnTaus"))),
      electronsToken_(consumes<std::vector<l1t::TkElectron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
      muonsToken_(consumes<std::vector<l1t::SAMuon>>(iConfig.getParameter<edm::InputTag>("muons"))),
      genJetsFlavourToken_(
          consumes<reco::JetFlavourInfoMatchingCollection>(iConfig.getParameter<edm::InputTag>("genJetsFlavour"))),
      vtxEmuToken_(consumes<std::vector<l1t::VertexWord>>(iConfig.getParameter<edm::InputTag>("vtx"))),
      bjetidsToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("bjetIDs"))),
      offlineJetsToken_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("offlineJets"))),
      offlinePVsToken_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("offlinePVs"))),
      doOfflineInfo_(iConfig.getParameter<bool>("doOfflineInfo")) {
  produces<nanoaod::FlatTable>("L1JetTrain");
  produces<nanoaod::FlatTable>("L1JetTrainCand");
  if (doOfflineInfo_)
    produces<nanoaod::FlatTable>("L1JetTrainOffCand");
  produces<nanoaod::FlatTable>("L1GenH");
  produces<nanoaod::FlatTable>("L1GenEvent");
}

l1jettraintools::GenEventInfo L1JetTrainingTableProducer::fillGenInfo(
    const edm::Event& iEvent, const edm::Handle<reco::GenParticleCollection>& genParticles) const {
  using namespace l1jettraintools;
  GenEventInfo info;
  if (iEvent.isRealData() || !genParticles.isValid())
    return info;
  info.isMC = true;

  // PNET-style gen particle chain (H/W/Z, prompt leptons, hard partons, taus)
  std::vector<float> gen_particle_pt, gen_particle_eta, gen_particle_phi, gen_particle_mass;
  std::vector<int> gen_particle_id;
  std::vector<unsigned int> gen_particle_status;
  std::vector<int> gen_particle_daughters_id;
  std::vector<unsigned int> gen_particle_daughters_igen, gen_particle_daughters_status;
  std::vector<float> gen_particle_daughters_pt, gen_particle_daughters_eta, gen_particle_daughters_phi,
      gen_particle_daughters_mass;
  std::vector<int> gen_particle_daughters_charge;

  TLorentzVector gen4HH;
  {
    unsigned int igen = 0;
    for (auto gens_iter = genParticles->begin(); gens_iter != genParticles->end(); ++gens_iter) {
      TLorentzVector gen4H;
      uint countH = 0;
      if ((abs(gens_iter->pdgId()) == 25) and gens_iter->isLastCopy() and gens_iter->statusFlags().fromHardProcess() and
          countH < 2) {
        countH++;
        gen4H.SetPtEtaPhiM(gens_iter->pt(), gens_iter->eta(), gens_iter->phi(), gens_iter->mass());
        gen4HH = gen4HH + gen4H;
        info.genH_mass.push_back(gen4H.M());
        info.genH_pt.push_back(gen4H.Pt());
        info.ngenH++;
      }

      if ((abs(gens_iter->pdgId()) == 25 or abs(gens_iter->pdgId()) == 24 or abs(gens_iter->pdgId()) == 23) and
          gens_iter->isLastCopy() and gens_iter->statusFlags().fromHardProcess()) {
        gen_particle_pt.push_back(gens_iter->pt());
        gen_particle_eta.push_back(gens_iter->eta());
        gen_particle_phi.push_back(gens_iter->phi());
        gen_particle_mass.push_back(gens_iter->mass());
        gen_particle_id.push_back(gens_iter->pdgId());
        gen_particle_status.push_back(gens_iter->status());
        for (size_t idau = 0; idau < gens_iter->numberOfDaughters(); idau++) {
          gen_particle_daughters_id.push_back(gens_iter->daughter(idau)->pdgId());
          gen_particle_daughters_igen.push_back(igen);
          gen_particle_daughters_pt.push_back(gens_iter->daughter(idau)->pt());
          gen_particle_daughters_eta.push_back(gens_iter->daughter(idau)->eta());
          gen_particle_daughters_phi.push_back(gens_iter->daughter(idau)->phi());
          gen_particle_daughters_mass.push_back(gens_iter->daughter(idau)->mass());
          gen_particle_daughters_status.push_back(gens_iter->daughter(idau)->status());
          gen_particle_daughters_charge.push_back(gens_iter->daughter(idau)->charge());
        }
        igen++;
      }

      // final state leptons (e,mu,nu), prompt or prompt tau decay products
      if (abs(gens_iter->pdgId()) > 10 and abs(gens_iter->pdgId()) < 17 and abs(gens_iter->pdgId()) != 15 and
          (gens_iter->isPromptFinalState() or gens_iter->isDirectPromptTauDecayProductFinalState())) {
        gen_particle_pt.push_back(gens_iter->pt());
        gen_particle_eta.push_back(gens_iter->eta());
        gen_particle_phi.push_back(gens_iter->phi());
        gen_particle_mass.push_back(gens_iter->mass());
        gen_particle_id.push_back(gens_iter->pdgId());
        gen_particle_status.push_back(gens_iter->status());
        igen++;
      }

      // hard-process partons, first copy
      if (((abs(gens_iter->pdgId()) >= 1 and abs(gens_iter->pdgId()) <= 5) or abs(gens_iter->pdgId()) == 21) and
          gens_iter->statusFlags().fromHardProcess() and gens_iter->statusFlags().isFirstCopy()) {
        gen_particle_pt.push_back(gens_iter->pt());
        gen_particle_eta.push_back(gens_iter->eta());
        gen_particle_phi.push_back(gens_iter->phi());
        gen_particle_mass.push_back(gens_iter->mass());
        gen_particle_id.push_back(gens_iter->pdgId());
        gen_particle_status.push_back(gens_iter->status());
        igen++;
      }

      // taus: last copy, from hard process, prompt decayed
      if (abs(gens_iter->pdgId()) == 15 and gens_iter->isLastCopy() and gens_iter->statusFlags().fromHardProcess() and
          gens_iter->isPromptDecayed()) {
        gen_particle_pt.push_back(gens_iter->pt());
        gen_particle_eta.push_back(gens_iter->eta());
        gen_particle_phi.push_back(gens_iter->phi());
        gen_particle_mass.push_back(gens_iter->mass());
        gen_particle_id.push_back(gens_iter->pdgId());
        gen_particle_status.push_back(gens_iter->status());
        for (size_t idau = 0; idau < gens_iter->numberOfDaughters(); idau++) {
          if (not dynamic_cast<const reco::GenParticle*>(gens_iter->daughter(idau))
                  ->statusFlags()
                  .isPromptTauDecayProduct())
            continue;
          gen_particle_daughters_id.push_back(gens_iter->daughter(idau)->pdgId());
          gen_particle_daughters_igen.push_back(igen);
          gen_particle_daughters_pt.push_back(gens_iter->daughter(idau)->pt());
          gen_particle_daughters_eta.push_back(gens_iter->daughter(idau)->eta());
          gen_particle_daughters_phi.push_back(gens_iter->daughter(idau)->phi());
          gen_particle_daughters_mass.push_back(gens_iter->daughter(idau)->mass());
          gen_particle_daughters_status.push_back(gens_iter->daughter(idau)->status());
          gen_particle_daughters_charge.push_back(gens_iter->daughter(idau)->charge());
        }
        igen++;
      }
    }
    info.genHH_mass = gen4HH.M();
    info.genHH_pt = gen4HH.Pt();
  }

  // gen leptons from resonance decays (H/ZZ/WW/tau -> e/mu)
  for (size_t igen = 0; igen < gen_particle_pt.size(); igen++) {
    if (abs(gen_particle_id.at(igen)) == 25 or abs(gen_particle_id.at(igen)) == 23 or abs(gen_particle_id.at(igen)) == 24 or
        abs(gen_particle_id.at(igen)) == 15) {
      for (size_t idau = 0; idau < gen_particle_daughters_id.size(); idau++) {
        if (gen_particle_daughters_igen.at(idau) == igen and
            (abs(gen_particle_daughters_id.at(idau)) == 11 or abs(gen_particle_daughters_id.at(idau)) == 13)) {
          TLorentzVector gen4V;
          gen4V.SetPtEtaPhiM(gen_particle_daughters_pt.at(idau),
                             gen_particle_daughters_eta.at(idau),
                             gen_particle_daughters_phi.at(idau),
                             gen_particle_daughters_mass.at(idau));
          if (abs(gen_particle_daughters_id.at(idau)) == 13 and
              std::find(info.genMuonsFromResonance4V.begin(), info.genMuonsFromResonance4V.end(), gen4V) ==
                  info.genMuonsFromResonance4V.end()) {
            info.genMuonsFromResonance4V.push_back(gen4V);
          }
          if (abs(gen_particle_daughters_id.at(idau)) == 11 and
              std::find(info.genElectronsFromResonance4V.begin(), info.genElectronsFromResonance4V.end(), gen4V) ==
                  info.genElectronsFromResonance4V.end()) {
            info.genElectronsFromResonance4V.push_back(gen4V);
          }
        }
      }
    }
  }

  // visible gen taus
  for (size_t igen = 0; igen < gen_particle_pt.size(); igen++) {
    if (abs(gen_particle_id.at(igen)) == 15) {
      TLorentzVector tau_gen_tmp;
      unsigned int tau_gen_nch_tmp(0), tau_gen_np0_tmp(0), tau_gen_nnh_tmp(0);
      for (size_t idau = 0; idau < gen_particle_daughters_pt.size(); idau++) {
        if (gen_particle_daughters_igen.at(idau) == igen and abs(gen_particle_daughters_id.at(idau)) != 11 and
            abs(gen_particle_daughters_id.at(idau)) != 13 and abs(gen_particle_daughters_id.at(idau)) != 12 and
            abs(gen_particle_daughters_id.at(idau)) != 14 and abs(gen_particle_daughters_id.at(idau)) != 16) {
          TLorentzVector tmp4V;
          tmp4V.SetPtEtaPhiM(gen_particle_daughters_pt.at(idau),
                             gen_particle_daughters_eta.at(idau),
                             gen_particle_daughters_phi.at(idau),
                             gen_particle_daughters_mass.at(idau));
          tau_gen_tmp += tmp4V;
          if (gen_particle_daughters_charge.at(idau) != 0 and gen_particle_daughters_status.at(idau) == 1)
            tau_gen_nch_tmp++;
          else if (gen_particle_daughters_charge.at(idau) == 0 and gen_particle_daughters_id.at(idau) == 111)
            tau_gen_np0_tmp++;
          else if (gen_particle_daughters_charge.at(idau) == 0 and gen_particle_daughters_id.at(idau) != 111)
            tau_gen_nnh_tmp++;
        }
      }
      if (tau_gen_tmp.Pt() > 0) {  // good hadronic tau
        info.tauGenVisible4V.push_back(tau_gen_tmp);
        tau_gen_tmp.SetPtEtaPhiM(
            gen_particle_pt.at(igen), gen_particle_eta.at(igen), gen_particle_phi.at(igen), gen_particle_mass.at(igen));
        info.tauGenCharge.push_back((gen_particle_id.at(igen) > 0) ? -1 : 1);
        info.tauGen4V.push_back(tau_gen_tmp);
        info.tauGenNch.push_back(tau_gen_nch_tmp);
        info.tauGenNp0.push_back(tau_gen_np0_tmp);
        info.tauGenNnh.push_back(tau_gen_nnh_tmp);
      }
    }
  }

  return info;
}

void L1JetTrainingTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace l1jettraintools;

  // --- jet-level column buffers (names == JetNTuplizer branch names) ---
  std::vector<float> v_jet_eta, v_jet_eta_phys, v_jet_phi, v_jet_phi_phys, v_jet_pt, v_jet_pt_log, v_jet_pt_log_std,
      v_jet_pt_phys, v_jet_mass_phys, v_jet_energy_phys;
  std::vector<float> v_jet_SC4NGJet_score_b, v_jet_SC4NGJet_score_charm, v_jet_SC4NGJet_score_light,
      v_jet_SC4NGJet_score_gluon, v_jet_SC4NGJet_score_taup, v_jet_SC4NGJet_score_taum, v_jet_SC4NGJet_score_muon,
      v_jet_SC4NGJet_score_electron, v_jet_SC4NGJet_score_regression;
  std::vector<float> v_jet_bjetscore, v_jet_tauscore, v_jet_taupt, v_jet_taueta, v_jet_taumatch_dR;
  std::vector<float> v_jet_eletkiso, v_jet_elepfiso, v_jet_elepuppiiso, v_jet_elematch_dR;
  std::vector<float> v_jet_muoniso, v_jet_muonmatch_dR;
  std::vector<uint8_t> v_jet_reject;
  std::vector<int32_t> v_jet_tauflav, v_jet_muflav, v_jet_elflav, v_jet_taudecaymode, v_jet_lepflav, v_jet_taucharge;
  std::vector<float> v_jet_genmatch_lep_pt, v_jet_genmatch_lep_vis_pt, v_jet_genmatch_lep_dR;
  std::vector<float> v_jet_genmatch_pt, v_jet_genmatch_eta, v_jet_genmatch_phi, v_jet_genmatch_mass, v_jet_genmatch_dR;
  std::vector<uint32_t> v_jet_genmatch_hflav;
  std::vector<int32_t> v_jet_genmatch_pflav;
  std::vector<float> v_jet_jecmatch_dR, v_jet_pt_corr;
  std::vector<int32_t> v_jet_npuppicand;
  // offline jet-level buffers
  std::vector<float> v_jet_offline_pt, v_jet_offline_genPt_, v_jet_offline_eta, v_jet_offline_phi, v_jet_offline_mass,
      v_jet_offline_dR;
  std::vector<uint32_t> v_jet_offline_hflav;
  std::vector<int32_t> v_jet_offline_pflav;
  std::vector<float> v_jet_offline_btag_deepcsv_probb, v_jet_offline_btag_deepcsv_probc, v_jet_offline_btag_deepcsv_probudsg,
      v_jet_offline_btag_deepjet_probb, v_jet_offline_btag_deepjet_probc, v_jet_offline_btag_deepjet_probuds,
      v_jet_offline_btag_deepjet_probg, v_jet_offline_btag_pnet_probb, v_jet_offline_btag_pnet_probc,
      v_jet_offline_btag_pnet_probuds, v_jet_offline_btag_pnet_probg, v_jet_offline_btag_pnet_probtaup,
      v_jet_offline_btag_pnet_probtaum, v_jet_offline_btag_pnet_probele, v_jet_offline_btag_pnet_probmu,
      v_jet_offline_btag_pnetForward_probq, v_jet_offline_btag_pnetForward_probg;
  std::vector<int32_t> v_njet_puppicand_off;

  // --- flattened candidate buffers (+ jetIdx back-references) ---
  std::vector<int32_t> c_jetIdx;
  std::vector<float> c_pt, c_eta, c_phi, c_pt_rel, c_pt_log, c_float_pt_log, c_deta, c_dphi, c_mass, c_isPhoton,
      c_isElectronPlus, c_isElectronMinus, c_isMuonPlus, c_isMuonMinus, c_isNeutralHadron, c_isChargedHadronPlus,
      c_isChargedHadronMinus, c_z0, c_dxy, c_isfilled, c_puppiweight, c_emid, c_quality, c_charge, c_id, c_eta_phys,
      c_phi_phys;
  // flattened offline-candidate buffers
  std::vector<int32_t> oc_jetIdx;
  std::vector<float> oc_pt, oc_pt_log, oc_pt_rel, oc_eta, oc_eta_phys, oc_deta, oc_deta_phys, oc_phi, oc_phi_phys,
      oc_dphi, oc_dphi_phys, oc_mass, oc_dxy, oc_dxy_phys, oc_z0, oc_z0_phys, oc_quality, oc_puppiweight, oc_isfilled,
      oc_emid, oc_charge, oc_id, oc_isPhoton, oc_isElectronPlus, oc_isElectronMinus, oc_isMuonPlus, oc_isMuonMinus,
      oc_isHadronPlus, oc_isHadronMinus, oc_isNeutralHadron;

  edm::Handle<std::vector<reco::GenJet>> genjets;
  edm::Handle<reco::GenParticleCollection> genparticles;
  edm::Handle<std::vector<l1t::PFJet>> scjets;
  edm::Handle<std::vector<l1t::PFJet>> scjetsCorr;
  edm::Handle<std::vector<l1t::PFTau>> nntaus;
  edm::Handle<std::vector<l1t::TkElectron>> electrons;
  edm::Handle<std::vector<l1t::SAMuon>> muons;
  edm::Handle<std::vector<pat::Jet>> offlineJets;
  edm::Handle<std::vector<reco::Vertex>> offlinePVs;
  edm::Handle<reco::JetFlavourInfoMatchingCollection> genJetsFlavour;

  const bool isMC = !iEvent.isRealData();
  GenEventInfo geninfo;
  if (isMC) {
    iEvent.getByToken(genparticlesToken_, genparticles);
    geninfo = fillGenInfo(iEvent, genparticles);
    iEvent.getByToken(genjetsToken_, genjets);
    iEvent.getByToken(genJetsFlavourToken_, genJetsFlavour);
  }
  iEvent.getByToken(scjetsToken_, scjets);
  iEvent.getByToken(scjetsCorrToken_, scjetsCorr);
  iEvent.getByToken(nntausToken_, nntaus);
  iEvent.getByToken(electronsToken_, electrons);
  iEvent.getByToken(muonsToken_, muons);
  if (doOfflineInfo_) {
    iEvent.getByToken(offlineJetsToken_, offlineJets);
    iEvent.getByToken(offlinePVsToken_, offlinePVs);
  }

  edm::Handle<edm::ValueMap<float>> bjetIDhandle;
  iEvent.getByToken(bjetidsToken_, bjetIDhandle);

  // gen jets sorted by pt (and genHT sums)
  std::vector<reco::GenJetRef> jetv_gen;
  if (genjets.isValid()) {
    for (auto jets_iter = genjets->begin(); jets_iter != genjets->end(); ++jets_iter) {
      reco::GenJetRef jref(genjets, jets_iter - genjets->begin());
      jetv_gen.push_back(jref);
      geninfo.genHTall += jref->pt();
      if (jref->pt() > 30. && fabs(jref->eta()) < 2.4) {
        geninfo.genHT += jref->pt();
      }
    }
    sort(jetv_gen.begin(), jetv_gen.end(), genJetRefSorter);
  }

  // reco jets (NG jets), selection identical to JetNTuplizer
  std::vector<l1t::PFJetRef> jetv_l1;
  for (auto jets_iter = scjets->begin(); jets_iter != scjets->end(); ++jets_iter) {
    l1t::PFJetRef jref(scjets, jets_iter - scjets->begin());
    if (jref->pt() < jetPtMin_)
      continue;
    if (fabs(jref->eta()) > jetEtaMax_)
      continue;
    if (fabs(jref->eta()) < jetEtaMin_)
      continue;
    jetv_l1.push_back(jref);
  }
  // for event level plots, always store something
  if (jetv_l1.size() < 1) {
    jetv_l1.clear();
    for (auto jets_iter = scjets->begin(); jets_iter != scjets->end(); ++jets_iter) {
      l1t::PFJetRef jref(scjets, jets_iter - scjets->begin());
      jetv_l1.push_back(jref);
    }
  }
  sort(jetv_l1.begin(), jetv_l1.end(), jetRefSorter);

  // reco jets with JEC, sorted by pt
  std::vector<l1t::PFJetRef> jetv_l1_corr;
  for (auto jets_iter_corr = scjetsCorr->begin(); jets_iter_corr != scjetsCorr->end(); ++jets_iter_corr) {
    l1t::PFJetRef jrefcorr(scjetsCorr, jets_iter_corr - scjetsCorr->begin());
    jetv_l1_corr.push_back(jrefcorr);
  }
  sort(jetv_l1_corr.begin(), jetv_l1_corr.end(), jetRefSorter);

  // reco taus with pt > 5
  std::vector<l1t::PFTauRef> tauv_l1;
  for (auto tau_iter = nntaus->begin(); tau_iter != nntaus->end(); ++tau_iter) {
    l1t::PFTauRef tref(nntaus, tau_iter - nntaus->begin());
    if (tref->pt() < 5.)
      continue;
    tauv_l1.push_back(tref);
  }
  sort(tauv_l1.begin(), tauv_l1.end(), tauRefSorter);

  // reco electrons with pt > 5
  std::vector<l1t::TkElectronRef> elev_l1;
  for (auto ele_iter = electrons->begin(); ele_iter != electrons->end(); ++ele_iter) {
    l1t::TkElectronRef eref(electrons, ele_iter - electrons->begin());
    if (eref->pt() < 5.)
      continue;
    elev_l1.push_back(eref);
  }
  sort(elev_l1.begin(), elev_l1.end(), eleRefSorter);

  // reco muons with pt > 5
  std::vector<l1t::SAMuonRef> muonv_l1;
  for (auto muon_iter = muons->begin(); muon_iter != muons->end(); ++muon_iter) {
    l1t::SAMuonRef mref(muons, muon_iter - muons->begin());
    if (mref->pt() < 5.)
      continue;
    muonv_l1.push_back(mref);
  }
  sort(muonv_l1.begin(), muonv_l1.end(), muonRefSorter);

  // offline jets sorted by pt
  std::vector<pat::JetRef> jetv_offline;
  if (doOfflineInfo_) {
    for (auto jets_iter = offlineJets->begin(); jets_iter != offlineJets->end(); ++jets_iter) {
      pat::JetRef jref(offlineJets, jets_iter - offlineJets->begin());
      jetv_offline.push_back(jref);
    }
    sort(jetv_offline.begin(), jetv_offline.end(), offlineJetRefSorter);
  }

  // --- loop over reco jets (one row per jet == one entry in JetNTuplizer tree) ---
  for (size_t i = 0; i < jetv_l1.size(); i++) {
    const auto& jetref = jetv_l1[i];
    l1ct::Jet ctJet = l1ct::Jet::unpack(jetref->getHWJetCT());
    std::vector<float> tagScores = jetref->getTagScores();
    float score_light = -999, score_b = -999, score_taup = -999, score_taum = -999, score_gluon = -999,
          score_charm = -999, score_muon = -999, score_electron = -999, score_regression = 1;
    if (tagScores.size() > 0) {
      score_b = tagScores[0];
      score_charm = tagScores[1];
      score_light = tagScores[2];
      score_gluon = tagScores[3];
      score_taup = tagScores[4];
      score_taum = tagScores[5];
      score_muon = tagScores[6];
      score_electron = tagScores[7];
      score_regression = jetref->getPtCorrection();
    }

    const float jet_pt = float(ctJet.hwPt);
    constexpr int LOG_LUT_SIZE = 256;
    const float jet_pt_log =
        L1TSC4NGJet::inputtype(l1ct::log_with_shift<l1ct::pt_t, L1TSC4NGJet::log_pt_t, LOG_LUT_SIZE>(ctJet.hwPt));
    const float jet_pt_log_std = std::log(float(ctJet.hwPt));
    const float jet_eta = L1TSC4NGJet::inputtype(ctJet.hwEta);
    const float jet_phi = float(ctJet.hwPhi);
    const float jet_pt_phys = jetref->pt();
    const float jet_eta_phys = jetref->eta();
    const float jet_phi_phys = jetref->phi();
    const float jet_mass_phys = jetref->mass();
    const float jet_energy_phys = jetref->energy();

    const float jet_bjetscore = (*bjetIDhandle)[jetref];

    // match to GEN jets
    float jet_genmatch_pt = 0, jet_genmatch_eta = 0, jet_genmatch_phi = 0, jet_genmatch_mass = 0, jet_genmatch_dR = 0;
    unsigned int jet_genmatch_hflav = 0;
    int jet_genmatch_pflav = 0;
    {
      int pos_matched = -1;
      float minDR = dRJetGenMatch_;
      for (size_t igen = 0; igen < jetv_gen.size(); igen++) {
        if (jetref->pt() <= 0.1 * jetv_gen[igen]->pt())
          continue;
        if (reco::deltaR(jetv_gen[igen]->p4(), jetref->p4()) < minDR) {
          pos_matched = igen;
          minDR = reco::deltaR(jetv_gen[igen]->p4(), jetref->p4());
        }
      }
      if (pos_matched >= 0) {
        jet_genmatch_pt = jetv_gen[pos_matched]->pt();
        jet_genmatch_eta = jetv_gen[pos_matched]->eta();
        jet_genmatch_phi = jetv_gen[pos_matched]->phi();
        jet_genmatch_mass = jetv_gen[pos_matched]->mass();
        jet_genmatch_dR = minDR;
        jet_genmatch_hflav =
            (*genJetsFlavour)[edm::RefToBase<reco::Jet>(jetv_gen[pos_matched])].getHadronFlavour();
        jet_genmatch_pflav =
            (*genJetsFlavour)[edm::RefToBase<reco::Jet>(jetv_gen[pos_matched])].getPartonFlavour();
      }
    }

    // match to gen leptons (muons/electrons/had taus) from resonance decays
    float minDR = 1000;
    int nlep_in_cone = 0;
    int pos_matched_genmu = -1;
    int pos_matched_genele = -1;
    int pos_matched_tauh = -1;
    int gentau_decaymode = -1;
    TLorentzVector genLepton4V, genLeptonVis4V;
    TLorentzVector jet4V;
    jet4V.SetPtEtaPhiM(jetref->pt(), jetref->eta(), jetref->phi(), jetref->mass());
    for (size_t igen = 0; igen < geninfo.genMuonsFromResonance4V.size(); igen++) {
      float dR = jet4V.DeltaR(geninfo.genMuonsFromResonance4V.at(igen));
      if (dR < dRCone_)
        nlep_in_cone++;
      if (dR < dRCone_ and dR < minDR) {
        pos_matched_genmu = igen;
        minDR = dR;
        genLepton4V = geninfo.genMuonsFromResonance4V.at(igen);
        genLeptonVis4V = geninfo.genMuonsFromResonance4V.at(igen);
      }
    }
    for (size_t igen = 0; igen < geninfo.genElectronsFromResonance4V.size(); igen++) {
      float dR = jet4V.DeltaR(geninfo.genElectronsFromResonance4V.at(igen));
      if (dR < dRCone_)
        nlep_in_cone++;
      if (dR < dRCone_ and dR < minDR) {
        pos_matched_genmu = -1;
        pos_matched_genele = igen;
        minDR = dR;
        genLepton4V = geninfo.genElectronsFromResonance4V.at(igen);
        genLeptonVis4V = geninfo.genElectronsFromResonance4V.at(igen);
      }
    }
    for (size_t itau = 0; itau < geninfo.tauGenVisible4V.size(); itau++) {
      float dR = geninfo.tauGenVisible4V.at(itau).DeltaR(jet4V);
      if (dR < dRCone_)
        nlep_in_cone++;
      if (dR < dRCone_ and dR < minDR) {
        pos_matched_genmu = -1;
        pos_matched_genele = -1;
        pos_matched_tauh = itau;
        minDR = dR;
        gentau_decaymode = 5 * (geninfo.tauGenNch.at(itau) - 1) + geninfo.tauGenNp0.at(itau);
        genLepton4V = geninfo.tauGen4V.at(itau);
        genLeptonVis4V = geninfo.tauGenVisible4V.at(itau);
      }
    }

    // basic selection criteria
    uint8_t jet_reject = false;
    if (applyJetIDFlag_ && pos_matched_genmu == -1 && pos_matched_genele == -1 && pos_matched_tauh == -1)
      jet_reject = true;
    if (pos_matched_genmu != -1 and genLeptonVis4V.Pt() < ptGenLeptonMin_)
      jet_reject = true;
    if (pos_matched_genele != -1 and genLeptonVis4V.Pt() < ptGenLeptonMin_)
      jet_reject = true;
    if (pos_matched_tauh != -1 and genLeptonVis4V.Pt() < ptGenTauVisibleMin_)
      jet_reject = true;

    // flavour info
    int jet_muflav = (pos_matched_genmu >= 0) ? 1 : 0;
    int jet_elflav = (pos_matched_genele >= 0) ? 1 : 0;
    int jet_tauflav, jet_taudecaymode, jet_taucharge;
    if (pos_matched_tauh >= 0) {
      jet_tauflav = 1;
      jet_taudecaymode = gentau_decaymode;
      jet_taucharge = geninfo.tauGenCharge.at(pos_matched_tauh);
    } else {
      jet_tauflav = 0;
      jet_taudecaymode = -1;
      jet_taucharge = 0;
    }
    const int jet_lepflav = nlep_in_cone;
    const float jet_genmatch_lep_pt = genLepton4V.Pt();
    const float jet_genmatch_lep_vis_pt = genLeptonVis4V.Pt();
    const float jet_genmatch_lep_dR = minDR;

    // match to L1 taus
    float jet_taumatch_dR = 999., jet_tauscore = -1., jet_taupt = -1., jet_taueta = -1.;
    {
      int pos_matched_tau = -1;
      float minDR_tau = 0.4;
      for (size_t itau = 0; itau < tauv_l1.size(); itau++) {
        if (reco::deltaR(tauv_l1[itau]->p4(), jetref->p4()) < minDR_tau) {
          pos_matched_tau = itau;
          minDR_tau = reco::deltaR(tauv_l1[itau]->p4(), jetref->p4());
        }
      }
      if (pos_matched_tau > -1) {
        jet_taumatch_dR = minDR_tau;
        jet_tauscore = tauv_l1[pos_matched_tau]->chargedIso();
        jet_taupt = tauv_l1[pos_matched_tau]->pt();
        jet_taueta = tauv_l1[pos_matched_tau]->eta();
      }
    }

    // match to L1 electrons
    float jet_elematch_dR = 999., jet_eletkiso = -1., jet_elepfiso = -1., jet_elepuppiiso = -1.;
    {
      int pos_matched_ele = -1;
      float minDR_ele = 0.4;
      for (size_t iele = 0; iele < elev_l1.size(); iele++) {
        if (reco::deltaR(elev_l1[iele]->p4(), jetref->p4()) < minDR_ele) {
          pos_matched_ele = iele;
          minDR_ele = reco::deltaR(elev_l1[iele]->p4(), jetref->p4());
        }
      }
      if (pos_matched_ele > -1) {
        jet_elematch_dR = minDR_ele;
        jet_eletkiso = elev_l1[pos_matched_ele]->trkIsol();
        jet_elepfiso = elev_l1[pos_matched_ele]->pfIsol();
        jet_elepuppiiso = elev_l1[pos_matched_ele]->puppiIsol();
      }
    }

    // match to L1 muons
    float jet_muonmatch_dR = 999., jet_muoniso = -1.;
    {
      int pos_matched_muon = -1;
      float minDR_muon = 0.4;
      for (size_t imuon = 0; imuon < muonv_l1.size(); imuon++) {
        if (reco::deltaR(muonv_l1[imuon]->p4(), jetref->p4()) < minDR_muon) {
          pos_matched_muon = imuon;
          minDR_muon = reco::deltaR(muonv_l1[imuon]->p4(), jetref->p4());
        }
      }
      if (pos_matched_muon > -1) {
        jet_muonmatch_dR = minDR_muon;
        jet_muoniso = muonv_l1[pos_matched_muon]->hwIso();
      }
    }

    // match to JEC-corrected jets
    float jet_jecmatch_dR = 999., jet_pt_corr = -1.;
    {
      int pos_matched_jec = -1;
      float minDR_jec = 0.4;
      for (size_t ijec = 0; ijec < jetv_l1_corr.size(); ijec++) {
        if (reco::deltaR(jetv_l1_corr[ijec]->p4(), jetref->p4()) < minDR_jec) {
          pos_matched_jec = ijec;
          minDR_jec = reco::deltaR(jetv_l1_corr[ijec]->p4(), jetref->p4());
        }
      }
      if (pos_matched_jec > -1) {
        jet_jecmatch_dR = minDR_jec;
        jet_pt_corr = jetv_l1_corr[pos_matched_jec]->pt();
      }
    }

    if (doOfflineInfo_) {
      // match to offline jets
      int pos_matched_offline = -1;
      float minDR_offline = 0.4;
      int njet_pfcand_off = 0;
      for (size_t ioff = 0; ioff < jetv_offline.size(); ioff++) {
        if (reco::deltaR(jetv_offline[ioff]->p4(), jetref->p4()) < minDR_offline) {
          pos_matched_offline = ioff;
          minDR_offline = reco::deltaR(jetv_offline[ioff]->p4(), jetref->p4());
        }
      }

      if (pos_matched_offline > -1) {
        const auto& offjet = jetv_offline[pos_matched_offline];
        v_jet_offline_dR.push_back(minDR_offline);
        v_jet_offline_pt.push_back(offjet->pt());
        v_jet_offline_hflav.push_back(abs(offjet->hadronFlavour()));
        v_jet_offline_pflav.push_back(offjet->partonFlavour());
        v_jet_offline_genPt_.push_back(offjet->genJet() ? offjet->genJet()->pt() : -1.);
        v_jet_offline_eta.push_back(offjet->eta());
        v_jet_offline_phi.push_back(offjet->phi());
        v_jet_offline_mass.push_back(offjet->mass());

        if (fabs(offjet->eta()) < 2.4) {
          v_jet_offline_btag_deepcsv_probb.push_back(offjet->bDiscriminator("pfDeepCSVJetTags:probb") +
                                                     offjet->bDiscriminator("pfDeepCSVJetTags:probbb"));
          v_jet_offline_btag_deepcsv_probc.push_back(offjet->bDiscriminator("pfDeepCSVJetTags:probc"));
          v_jet_offline_btag_deepcsv_probudsg.push_back(offjet->bDiscriminator("pfDeepCSVJetTags:probudsg"));
          v_jet_offline_btag_deepjet_probb.push_back(offjet->bDiscriminator("pfDeepFlavourJetTags:probb") +
                                                     offjet->bDiscriminator("pfDeepFlavourJetTags:probbb") +
                                                     offjet->bDiscriminator("pfDeepFlavourJetTags:problepb"));
          v_jet_offline_btag_deepjet_probc.push_back(offjet->bDiscriminator("pfDeepFlavourJetTags:probc"));
          v_jet_offline_btag_deepjet_probuds.push_back(offjet->bDiscriminator("pfDeepFlavourJetTags:probuds"));
          v_jet_offline_btag_deepjet_probg.push_back(offjet->bDiscriminator("pfDeepFlavourJetTags:probg"));
          v_jet_offline_btag_pnet_probb.push_back(offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probb"));
          v_jet_offline_btag_pnet_probc.push_back(offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probc"));
          v_jet_offline_btag_pnet_probuds.push_back(offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probuds"));
          v_jet_offline_btag_pnet_probg.push_back(offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probg"));
          v_jet_offline_btag_pnet_probmu.push_back(offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probmu"));
          v_jet_offline_btag_pnet_probele.push_back(offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probele"));
          v_jet_offline_btag_pnet_probtaup.push_back(
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup1h0p") +
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup1h1p") +
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup1h2p") +
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup3h0p") +
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaup3h1p"));
          v_jet_offline_btag_pnet_probtaum.push_back(
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum1h0p") +
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum1h1p") +
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum1h2p") +
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum3h0p") +
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probtaum3h1p"));
        } else {
          v_jet_offline_btag_deepcsv_probb.push_back(-1.);
          v_jet_offline_btag_deepcsv_probc.push_back(-1.);
          v_jet_offline_btag_deepcsv_probudsg.push_back(-1.);
          v_jet_offline_btag_deepjet_probb.push_back(-1.);
          v_jet_offline_btag_deepjet_probc.push_back(-1.);
          v_jet_offline_btag_deepjet_probuds.push_back(-1.);
          v_jet_offline_btag_deepjet_probg.push_back(-1.);
          v_jet_offline_btag_pnet_probb.push_back(-1.);
          v_jet_offline_btag_pnet_probc.push_back(-1.);
          v_jet_offline_btag_pnet_probuds.push_back(-1.);
          v_jet_offline_btag_pnet_probg.push_back(-1.);
          v_jet_offline_btag_pnet_probmu.push_back(-1.);
          v_jet_offline_btag_pnet_probele.push_back(-1.);
          v_jet_offline_btag_pnet_probtaup.push_back(-1.);
          v_jet_offline_btag_pnet_probtaum.push_back(-1.);
        }

        if (fabs(offjet->eta()) > 2.4) {
          v_jet_offline_btag_pnetForward_probq.push_back(
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:probq"));
          v_jet_offline_btag_pnetForward_probg.push_back(
              offjet->bDiscriminator("pfParticleNetFromMiniAODAK4PuppiForwardJetTags:probg"));
        } else {
          v_jet_offline_btag_pnetForward_probq.push_back(-1.);
          v_jet_offline_btag_pnetForward_probg.push_back(-1.);
        }

        // offline candidate features of the matched offline jet
        TVector3 jet_direction_off(
            offjet->momentum().Unit().x(), offjet->momentum().Unit().y(), offjet->momentum().Unit().z());
        for (unsigned int j = 0; j < offjet->numberOfDaughters(); j++) {
          const reco::Candidate* daughter = offjet->daughter(j);
          const pat::PackedCandidate* packed = dynamic_cast<const pat::PackedCandidate*>(daughter);
          if (packed) {
            njet_pfcand_off++;
            oc_jetIdx.push_back(i);
            oc_pt.push_back(packed->pt());
            oc_pt_log.push_back(std::log(packed->pt()));
            oc_pt_rel.push_back(packed->pt() / offjet->pt());
            oc_eta.push_back(packed->eta() / (TMath::Pi() / 720.));  // l1ct::Scales::ETA_LSB
            oc_eta_phys.push_back(packed->eta());
            oc_deta.push_back((offjet->eta() - packed->eta()) / (TMath::Pi() / 720.));
            oc_deta_phys.push_back(offjet->eta() - packed->eta());
            TVector3 pfcand_momentum_off(packed->momentum().x(), packed->momentum().y(), packed->momentum().z());
            oc_dphi.push_back(jet_direction_off.DeltaPhi(pfcand_momentum_off) / (TMath::Pi() / 720.));
            oc_dphi_phys.push_back(jet_direction_off.DeltaPhi(pfcand_momentum_off));
            oc_phi.push_back(packed->phi() / (TMath::Pi() / 720.));
            oc_phi_phys.push_back(packed->phi());
            oc_mass.push_back(packed->mass());
            oc_dxy_phys.push_back(sqrt(abs(packed->dxy())));
            oc_dxy.push_back(int(sqrt(abs(packed->dxy())) / l1ct::Scales::DXYSQRT_LSB));
            oc_z0.push_back(packed->dz());
            oc_z0_phys.push_back(int(packed->dz() / l1ct::Scales::Z0_LSB));
            oc_quality.push_back(packed->hasTrackDetails() ? packed->pseudoTrack().qualityMask()
                                                           : (1 << reco::TrackBase::loose));
            oc_puppiweight.push_back(packed->puppiWeight());
            oc_isfilled.push_back(1.0);
            oc_emid.push_back(0.);
            oc_charge.push_back(packed->charge());
            oc_id.push_back(packed->pdgId());
            oc_isPhoton.push_back((fabs(packed->pdgId()) == 22));
            oc_isElectronPlus.push_back((packed->pdgId() == 11));
            oc_isElectronMinus.push_back((packed->pdgId() == -11));
            oc_isMuonPlus.push_back((packed->pdgId() == 13));
            oc_isMuonMinus.push_back((packed->pdgId() == -13));
            if ((abs(packed->pdgId()) != 22) && (abs(packed->pdgId()) != 11) && (abs(packed->pdgId()) != 13)) {
              if (packed->charge() != 0) {  // not neutral
                if ((fabs(packed->pdgId()) < 11) || (fabs(packed->pdgId()) > 13)) {
                  // not leptons -> hadrons
                  oc_isNeutralHadron.push_back(0);
                  oc_isHadronPlus.push_back((packed->charge() > 0));
                  oc_isHadronMinus.push_back((packed->charge() < 0));
                } else {
                  edm::LogWarning("L1JetTrainingTableProducer") << "charged lepton in jet constituents, pdgId "
                                                                << packed->pdgId() << " charge " << packed->charge();
                  oc_isNeutralHadron.push_back(0);
                  oc_isHadronPlus.push_back(0);
                  oc_isHadronMinus.push_back(0);
                }
              } else {  // neutral, exclude only photons
                oc_isNeutralHadron.push_back((fabs(packed->pdgId()) != 22));
                oc_isHadronPlus.push_back(0);
                oc_isHadronMinus.push_back(0);
              }
            } else {
              oc_isNeutralHadron.push_back(0);
              oc_isHadronPlus.push_back(0);
              oc_isHadronMinus.push_back(0);
            }
          }
        }
      } else {  // no offline match
        v_jet_offline_dR.push_back(999.);
        v_jet_offline_pt.push_back(-999.);
        v_jet_offline_hflav.push_back(999);
        v_jet_offline_pflav.push_back(999);
        v_jet_offline_genPt_.push_back(-1);
        v_jet_offline_eta.push_back(-999.);
        v_jet_offline_phi.push_back(-999.);
        v_jet_offline_mass.push_back(-999.);
        v_jet_offline_btag_deepcsv_probb.push_back(-1.);
        v_jet_offline_btag_deepcsv_probc.push_back(-1.);
        v_jet_offline_btag_deepcsv_probudsg.push_back(-1.);
        v_jet_offline_btag_deepjet_probb.push_back(-1.);
        v_jet_offline_btag_deepjet_probc.push_back(-1.);
        v_jet_offline_btag_deepjet_probuds.push_back(-1.);
        v_jet_offline_btag_deepjet_probg.push_back(-1.);
        v_jet_offline_btag_pnet_probb.push_back(-1.);
        v_jet_offline_btag_pnet_probc.push_back(-1.);
        v_jet_offline_btag_pnet_probuds.push_back(-1.);
        v_jet_offline_btag_pnet_probg.push_back(-1.);
        v_jet_offline_btag_pnet_probmu.push_back(-1.);
        v_jet_offline_btag_pnet_probele.push_back(-1.);
        v_jet_offline_btag_pnet_probtaup.push_back(-1.);
        v_jet_offline_btag_pnet_probtaum.push_back(-1.);
        v_jet_offline_btag_pnetForward_probq.push_back(-1.);
        v_jet_offline_btag_pnetForward_probg.push_back(-1.);
      }
      v_njet_puppicand_off.push_back(njet_pfcand_off);
    }  // if(doOfflineInfo_)

    // ============ push all jet-level values into the column buffers ============
    v_jet_eta.push_back(jet_eta);
    v_jet_eta_phys.push_back(jet_eta_phys);
    v_jet_phi.push_back(jet_phi);
    v_jet_phi_phys.push_back(jet_phi_phys);
    v_jet_pt.push_back(jet_pt);
    v_jet_pt_log.push_back(jet_pt_log);
    v_jet_pt_log_std.push_back(jet_pt_log_std);
    v_jet_pt_phys.push_back(jet_pt_phys);
    v_jet_mass_phys.push_back(jet_mass_phys);
    v_jet_energy_phys.push_back(jet_energy_phys);
    v_jet_SC4NGJet_score_b.push_back(score_b);
    v_jet_SC4NGJet_score_charm.push_back(score_charm);
    v_jet_SC4NGJet_score_light.push_back(score_light);
    v_jet_SC4NGJet_score_gluon.push_back(score_gluon);
    v_jet_SC4NGJet_score_taup.push_back(score_taup);
    v_jet_SC4NGJet_score_taum.push_back(score_taum);
    v_jet_SC4NGJet_score_muon.push_back(score_muon);
    v_jet_SC4NGJet_score_electron.push_back(score_electron);
    v_jet_SC4NGJet_score_regression.push_back(score_regression);
    v_jet_bjetscore.push_back(jet_bjetscore);
    v_jet_tauscore.push_back(jet_tauscore);
    v_jet_taupt.push_back(jet_taupt);
    v_jet_taueta.push_back(jet_taueta);
    v_jet_taumatch_dR.push_back(jet_taumatch_dR);
    v_jet_eletkiso.push_back(jet_eletkiso);
    v_jet_elepfiso.push_back(jet_elepfiso);
    v_jet_elepuppiiso.push_back(jet_elepuppiiso);
    v_jet_elematch_dR.push_back(jet_elematch_dR);
    v_jet_muoniso.push_back(jet_muoniso);
    v_jet_muonmatch_dR.push_back(jet_muonmatch_dR);
    v_jet_reject.push_back(jet_reject);
    v_jet_tauflav.push_back(jet_tauflav);
    v_jet_muflav.push_back(jet_muflav);
    v_jet_elflav.push_back(jet_elflav);
    v_jet_taudecaymode.push_back(jet_taudecaymode);
    v_jet_lepflav.push_back(jet_lepflav);
    v_jet_taucharge.push_back(jet_taucharge);
    v_jet_genmatch_lep_pt.push_back(jet_genmatch_lep_pt);
    v_jet_genmatch_lep_vis_pt.push_back(jet_genmatch_lep_vis_pt);
    v_jet_genmatch_lep_dR.push_back(jet_genmatch_lep_dR);
    v_jet_genmatch_pt.push_back(jet_genmatch_pt);
    v_jet_genmatch_eta.push_back(jet_genmatch_eta);
    v_jet_genmatch_phi.push_back(jet_genmatch_phi);
    v_jet_genmatch_mass.push_back(jet_genmatch_mass);
    v_jet_genmatch_dR.push_back(jet_genmatch_dR);
    v_jet_genmatch_hflav.push_back(jet_genmatch_hflav);
    v_jet_genmatch_pflav.push_back(jet_genmatch_pflav);
    v_jet_jecmatch_dR.push_back(jet_jecmatch_dR);
    v_jet_pt_corr.push_back(jet_pt_corr);

    // ============ puppi candidates of this jet (L1) ============
    std::vector<l1t::PFCandidate> vectorOfConstituents;
    for (unsigned ipart = 0; ipart < jetref->numberOfDaughters(); ipart++) {
      const l1t::PFCandidate* pfPart = dynamic_cast<const l1t::PFCandidate*>(jetref->daughter(ipart));
      vectorOfConstituents.push_back(*pfPart);
    }
    std::stable_sort(vectorOfConstituents.begin(), vectorOfConstituents.end(), l1PFCandidateSorter);

    int njet_pfcand = 0;
    for (size_t ipfcand = 0; ipfcand < vectorOfConstituents.size(); ipfcand++) {
      l1t::PFCandidate pfcand = vectorOfConstituents.at(ipfcand);
      if (pfcand.pt() < jetPFCandidatePtMin_)
        continue;
      njet_pfcand++;

      l1ct::PuppiObj puppicand = l1ct::PuppiObj::unpack(pfcand.encodedPuppi64());

      constexpr int INV_LUT_SIZE = 1024;
      L1TSC4NGJet::inputtype inv_jet_pt =
          L1TSC4NGJet::inputtype(l1ct::invert_with_shift<l1ct::pt_t, L1TSC4NGJet::inv_pt_t, INV_LUT_SIZE>(jet_pt));

      L1TSC4NGJet::inputtype log_cand_pt =
          L1TSC4NGJet::inputtype(l1ct::log_with_shift<l1ct::pt_t, L1TSC4NGJet::log_pt_t, LOG_LUT_SIZE>(puppicand.hwPt));

      L1SCJetEmu::detaphi_t puppi_dphi(puppicand.hwPhi - jet_phi);
      // phi wrap
      L1SCJetEmu::detaphi_t puppi_dphi0 = puppi_dphi > L1SCJetEmu::detaphi_t(l1ct::Scales::INTPHI_PI)
                                              ? L1SCJetEmu::detaphi_t(puppi_dphi - l1ct::Scales::INTPHI_TWOPI)
                                              : L1SCJetEmu::detaphi_t(puppi_dphi);
      L1SCJetEmu::detaphi_t puppi_dphi1 = puppi_dphi < L1SCJetEmu::detaphi_t(-l1ct::Scales::INTPHI_PI)
                                              ? L1SCJetEmu::detaphi_t(l1ct::Scales::INTPHI_TWOPI + puppi_dphi)
                                              : L1SCJetEmu::detaphi_t(puppi_dphi);
      L1TSC4NGJet::inputtype puppi_dphiw = puppi_dphi > L1SCJetEmu::detaphi_t(0) ? puppi_dphi0 : puppi_dphi1;

      L1TSC4NGJet::inputtype puppi_massCand = L1TSC4NGJet::candidate_mass<L1TSC4NGJet::inputtype>(puppicand);

      c_jetIdx.push_back(i);
      c_pt.push_back(puppicand.hwPt);
      c_pt_rel.push_back(L1TSC4NGJet::inputtype(puppicand.hwPt) * inv_jet_pt);
      c_pt_log.push_back(log_cand_pt);
      c_float_pt_log.push_back(std::log(float(puppicand.hwPt)));
      c_eta.push_back(puppicand.hwEta);
      c_phi.push_back(puppicand.hwPhi);
      c_deta.push_back(L1TSC4NGJet::inputtype(jet_eta) - L1TSC4NGJet::inputtype(puppicand.hwEta));
      c_dphi.push_back(puppi_dphiw);
      c_mass.push_back(puppi_massCand);
      c_isPhoton.push_back(puppicand.hwId.bits == l1ct::ParticleID::PHOTON);
      c_isElectronPlus.push_back(puppicand.hwId.bits == l1ct::ParticleID::ELEPLUS);
      c_isElectronMinus.push_back(puppicand.hwId.bits == l1ct::ParticleID::ELEMINUS);
      c_isMuonPlus.push_back(puppicand.hwId.bits == l1ct::ParticleID::MUPLUS);
      c_isMuonMinus.push_back(puppicand.hwId.bits == l1ct::ParticleID::MUMINUS);
      c_isNeutralHadron.push_back(puppicand.hwId.bits == l1ct::ParticleID::HADZERO);
      c_isChargedHadronPlus.push_back(puppicand.hwId.bits == l1ct::ParticleID::HADPLUS);
      c_isChargedHadronMinus.push_back(puppicand.hwId.bits == l1ct::ParticleID::HADMINUS);
      c_z0.push_back(L1TSC4NGJet::inputtype(puppicand.hwId.charged()) ? L1TSC4NGJet::inputtype(puppicand.hwZ0())
                                                                     : L1TSC4NGJet::inputtype(0.0));
      c_dxy.push_back(L1TSC4NGJet::inputtype(puppicand.hwId.charged()) ? L1TSC4NGJet::inputtype(puppicand.hwDxy())
                                                                      : L1TSC4NGJet::inputtype(0.0));
      c_isfilled.push_back(L1TSC4NGJet::inputtype(1.0));
      c_puppiweight.push_back(L1TSC4NGJet::inputtype(puppicand.hwId.neutral()) ? L1TSC4NGJet::inputtype(puppicand.hwPuppiW())
                                                                               : L1TSC4NGJet::inputtype(0.0));
      c_emid.push_back(L1TSC4NGJet::inputtype(puppicand.hwId.neutral()) ? L1TSC4NGJet::inputtype(puppicand.hwEmID())
                                                                        : L1TSC4NGJet::inputtype(0.0));
      c_quality.push_back(L1TSC4NGJet::inputtype(puppicand.hwId.charged()) ? L1TSC4NGJet::inputtype(puppicand.hwTkQuality())
                                                                           : L1TSC4NGJet::inputtype(0.0));
      c_charge.push_back(L1TSC4NGJet::inputtype(puppicand.hwId.charged()));
      c_id.push_back(L1TSC4NGJet::inputtype(puppicand.hwId.bits));
      c_eta_phys.push_back(pfcand.eta());
      c_phi_phys.push_back(pfcand.phi());
    }
    v_jet_npuppicand.push_back(njet_pfcand);
  }  // end loop over jets

  // ============ build FlatTables ============
  auto jetTab = std::make_unique<nanoaod::FlatTable>(jetv_l1.size(), "L1JetTrain", false);
  auto addF = [&jetTab](const std::string& name, const std::vector<float>& v, const std::string& doc) {
    jetTab->addColumn<float>(name, v, doc);
  };
  auto addI = [&jetTab](const std::string& name, const std::vector<int32_t>& v, const std::string& doc) {
    jetTab->addColumn<int32_t>(name, v, doc);
  };
  auto addU8 = [&jetTab](const std::string& name, const std::vector<uint8_t>& v, const std::string& doc) {
    jetTab->addColumn<uint8_t>(name, v, doc);
  };
  auto addU32 = [&jetTab](const std::string& name, const std::vector<uint32_t>& v, const std::string& doc) {
    jetTab->addColumn<uint32_t>(name, v, doc);
  };

  addF("jet_eta", v_jet_eta, "jet eta (hw units) = JetNTuplizer branch jet_eta");
  addF("jet_eta_phys", v_jet_eta_phys, "jet eta (physical)");
  addF("jet_phi", v_jet_phi, "jet phi (hw units)");
  addF("jet_phi_phys", v_jet_phi_phys, "jet phi (physical)");
  addF("jet_pt", v_jet_pt, "jet raw hw pt as float");
  addF("jet_pt_log", v_jet_pt_log, "log(hw pt) via LUT (NN input convention)");
  addF("jet_pt_log_std", v_jet_pt_log_std, "std::log(hw jet pt)");
  addF("jet_pt_phys", v_jet_pt_phys, "jet pt, physical (raw, returnRawPt collection)");
  addF("jet_mass_phys", v_jet_mass_phys, "jet mass");
  addF("jet_energy_phys", v_jet_energy_phys, "jet energy");
  addF("jet_SC4NGJet_score_b", v_jet_SC4NGJet_score_b, "SC4NGJet tag score: b");
  addF("jet_SC4NGJet_score_charm", v_jet_SC4NGJet_score_charm, "SC4NGJet tag score: c");
  addF("jet_SC4NGJet_score_light", v_jet_SC4NGJet_score_light, "SC4NGJet tag score: uds");
  addF("jet_SC4NGJet_score_gluon", v_jet_SC4NGJet_score_gluon, "SC4NGJet tag score: g");
  addF("jet_SC4NGJet_score_taup", v_jet_SC4NGJet_score_taup, "SC4NGJet tag score: tau+");
  addF("jet_SC4NGJet_score_taum", v_jet_SC4NGJet_score_taum, "SC4NGJet tag score: tau-");
  addF("jet_SC4NGJet_score_muon", v_jet_SC4NGJet_score_muon, "SC4NGJet tag score: mu");
  addF("jet_SC4NGJet_score_electron", v_jet_SC4NGJet_score_electron, "SC4NGJet tag score: e");
  addF("jet_SC4NGJet_score_regression", v_jet_SC4NGJet_score_regression, "SC4NGJet pt regression (correction factor)");
  addF("jet_bjetscore", v_jet_bjetscore, "L1 BJet NN score (from dedicated training bjet producer)");
  addF("jet_tauscore", v_jet_tauscore, "chargedIso of matched NNTau (dR<0.4), -1 if none");
  addF("jet_taupt", v_jet_taupt, "pt of matched NNTau");
  addF("jet_taueta", v_jet_taueta, "eta of matched NNTau");
  addF("jet_taumatch_dR", v_jet_taumatch_dR, "dR to matched NNTau, 999 if none");
  addF("jet_eletkiso", v_jet_eletkiso, "trkIsol of matched L1TkElectron, -1 if none");
  addF("jet_elepfiso", v_jet_elepfiso, "pfIsol of matched L1TkElectron");
  addF("jet_elepuppiiso", v_jet_elepuppiiso, "puppiIsol of matched L1TkElectron");
  addF("jet_elematch_dR", v_jet_elematch_dR, "dR to matched L1TkElectron, 999 if none");
  addF("jet_muoniso", v_jet_muoniso, "hwIso of matched L1SAMuon, -1 if none");
  addF("jet_muonmatch_dR", v_jet_muonmatch_dR, "dR to matched L1SAMuon, 999 if none");
  addU8("jet_reject", v_jet_reject, "JetNTuplizer jet_reject flag (0/1)");
  addI("jet_tauflav", v_jet_tauflav, "matched to hadronic gen tau (dR<0.2)");
  addI("jet_muflav", v_jet_muflav, "matched to gen muon from resonance (dR<0.2)");
  addI("jet_elflav", v_jet_elflav, "matched to gen electron from resonance (dR<0.2)");
  addI("jet_taudecaymode", v_jet_taudecaymode, "gen tau decay mode 5*(nch-1)+np0, -1 if no tau match");
  addI("jet_lepflav", v_jet_lepflav, "number of gen leptons in dR<0.2 cone");
  addI("jet_taucharge", v_jet_taucharge, "charge of matched gen tau");
  addF("jet_genmatch_lep_pt", v_jet_genmatch_lep_pt, "pt of matched gen lepton (0 if none)");
  addF("jet_genmatch_lep_vis_pt", v_jet_genmatch_lep_vis_pt, "visible pt of matched gen lepton");
  addF("jet_genmatch_lep_dR", v_jet_genmatch_lep_dR, "dR of gen lepton match, 1000 if none");
  addF("jet_genmatch_pt", v_jet_genmatch_pt, "matched gen jet pt (0 if none)");
  addF("jet_genmatch_eta", v_jet_genmatch_eta, "matched gen jet eta");
  addF("jet_genmatch_phi", v_jet_genmatch_phi, "matched gen jet phi");
  addF("jet_genmatch_mass", v_jet_genmatch_mass, "matched gen jet mass");
  addF("jet_genmatch_dR", v_jet_genmatch_dR, "dR of gen jet match (0 if none)");
  addU32("jet_genmatch_hflav", v_jet_genmatch_hflav, "hadron flavour of matched gen jet");
  addI("jet_genmatch_pflav", v_jet_genmatch_pflav, "parton flavour of matched gen jet");
  addF("jet_jecmatch_dR", v_jet_jecmatch_dR, "dR to matched corrected (JEC) jet, 999 if none");
  addF("jet_pt_corr", v_jet_pt_corr, "pt of matched corrected jet, -1 if none");
  addI("jet_npuppicand", v_jet_npuppicand, "number of puppi candidates of this jet (see L1JetTrainCand table)");
  if (doOfflineInfo_) {
    addF("jet_offline_pt", v_jet_offline_pt, "matched offline jet pt (-999 if none)");
    addU32("jet_offline_hflav", v_jet_offline_hflav, "hadron flavour of matched offline jet");
    addI("jet_offline_pflav", v_jet_offline_pflav, "parton flavour of matched offline jet");
    addF("jet_offline_genPt_", v_jet_offline_genPt_, "pt of genJet matched to offline jet");
    addF("jet_offline_eta", v_jet_offline_eta, "matched offline jet eta");
    addF("jet_offline_phi", v_jet_offline_phi, "matched offline jet phi");
    addF("jet_offline_mass", v_jet_offline_mass, "matched offline jet mass");
    addF("jet_offline_dR", v_jet_offline_dR, "dR to matched offline jet (999 if none)");
    addF("jet_offline_btag_deepcsv_probb", v_jet_offline_btag_deepcsv_probb, "DeepCSV probb+probbb");
    addF("jet_offline_btag_deepcsv_probc", v_jet_offline_btag_deepcsv_probc, "DeepCSV probc");
    addF("jet_offline_btag_deepcsv_probudsg", v_jet_offline_btag_deepcsv_probudsg, "DeepCSV probudsg");
    addF("jet_offline_btag_deepjet_probb", v_jet_offline_btag_deepjet_probb, "DeepJet probb+probbb+problepb");
    addF("jet_offline_btag_deepjet_probc", v_jet_offline_btag_deepjet_probc, "DeepJet probc");
    addF("jet_offline_btag_deepjet_probuds", v_jet_offline_btag_deepjet_probuds, "DeepJet probuds");
    addF("jet_offline_btag_deepjet_probg", v_jet_offline_btag_deepjet_probg, "DeepJet probg");
    addF("jet_offline_btag_pnet_probb", v_jet_offline_btag_pnet_probb, "PNet probb");
    addF("jet_offline_btag_pnet_probc", v_jet_offline_btag_pnet_probc, "PNet probc");
    addF("jet_offline_btag_pnet_probuds", v_jet_offline_btag_pnet_probuds, "PNet probuds");
    addF("jet_offline_btag_pnet_probg", v_jet_offline_btag_pnet_probg, "PNet probg");
    addF("jet_offline_btag_pnet_probtaup", v_jet_offline_btag_pnet_probtaup, "PNet probtaup (all had. decays)");
    addF("jet_offline_btag_pnet_probtaum", v_jet_offline_btag_pnet_probtaum, "PNet probtaum (all had. decays)");
    addF("jet_offline_btag_pnet_probele", v_jet_offline_btag_pnet_probele, "PNet probele");
    addF("jet_offline_btag_pnet_probmu", v_jet_offline_btag_pnet_probmu, "PNet probmu");
    addF("jet_offline_btag_pnetForward_probq", v_jet_offline_btag_pnetForward_probq, "PNet forward probq");
    addF("jet_offline_btag_pnetForward_probg", v_jet_offline_btag_pnetForward_probg, "PNet forward probg");
    addI("njet_puppicand_off", v_njet_puppicand_off, "number of offline candidates (see L1JetTrainOffCand table)");
  }
  iEvent.put(std::move(jetTab), "L1JetTrain");

  auto candTab = std::make_unique<nanoaod::FlatTable>(c_jetIdx.size(), "L1JetTrainCand", false);
  auto addFC = [&candTab](const std::string& name, const std::vector<float>& v, const std::string& doc) {
    candTab->addColumn<float>(name, v, doc);
  };
  candTab->addColumn<int32_t>("jetIdx", c_jetIdx, "index of parent jet in L1JetTrain table");
  addFC("jet_puppicand_pt", c_pt, "puppi candidate hw pt");
  addFC("jet_puppicand_eta", c_eta, "puppi candidate hw eta");
  addFC("jet_puppicand_phi", c_phi, "puppi candidate hw phi");
  addFC("jet_puppicand_pt_rel", c_pt_rel, "cand hw pt * inv(jet hw pt) (LUT)");
  addFC("jet_puppicand_pt_log", c_pt_log, "log(cand hw pt) via LUT");
  addFC("jet_puppicand_float_pt_log", c_float_pt_log, "std::log(cand hw pt)");
  addFC("jet_puppicand_deta", c_deta, "jet hw eta - cand hw eta");
  addFC("jet_puppicand_dphi", c_dphi, "wrapped dphi(cand,jet), hw units");
  addFC("jet_puppicand_mass", c_mass, "cand mass by PID");
  addFC("jet_puppicand_isPhoton", c_isPhoton, "hwId == PHOTON");
  addFC("jet_puppicand_isElectronPlus", c_isElectronPlus, "hwId == ELEPLUS");
  addFC("jet_puppicand_isElectronMinus", c_isElectronMinus, "hwId == ELEMINUS");
  addFC("jet_puppicand_isMuonPlus", c_isMuonPlus, "hwId == MUPLUS");
  addFC("jet_puppicand_isMuonMinus", c_isMuonMinus, "hwId == MUMINUS");
  addFC("jet_puppicand_isNeutralHadron", c_isNeutralHadron, "hwId == HADZERO");
  addFC("jet_puppicand_isChargedHadronPlus", c_isChargedHadronPlus, "hwId == HADPLUS");
  addFC("jet_puppicand_isChargedHadronMinus", c_isChargedHadronMinus, "hwId == HADMINUS");
  addFC("jet_puppicand_z0", c_z0, "z0 (charged only, else 0)");
  addFC("jet_puppicand_dxy", c_dxy, "dxy (charged only, else 0)");
  addFC("jet_puppicand_isfilled", c_isfilled, "always 1 (NN feature convention)");
  addFC("jet_puppicand_puppiweight", c_puppiweight, "puppi weight (neutral only, else 0)");
  addFC("jet_puppicand_emid", c_emid, "EM ID (neutral only, else 0)");
  addFC("jet_puppicand_quality", c_quality, "track quality (charged only, else 0)");
  addFC("jet_puppicand_charge", c_charge, "charged flag (hwId.charged())");
  addFC("jet_puppicand_id", c_id, "hwId bits");
  addFC("jet_puppicand_eta_phys", c_eta_phys, "physical cand eta");
  addFC("jet_puppicand_phi_phys", c_phi_phys, "physical cand phi");
  iEvent.put(std::move(candTab), "L1JetTrainCand");

  if (doOfflineInfo_) {
    auto offTab = std::make_unique<nanoaod::FlatTable>(oc_jetIdx.size(), "L1JetTrainOffCand", false);
    auto addFO = [&offTab](const std::string& name, const std::vector<float>& v, const std::string& doc) {
      offTab->addColumn<float>(name, v, doc);
    };
    offTab->addColumn<int32_t>("jetIdx", oc_jetIdx, "index of parent jet in L1JetTrain table");
    addFO("jet_offline_puppicand_pt", oc_pt, "offline cand pt");
    addFO("jet_offline_puppicand_pt_log", oc_pt_log, "log(offline cand pt)");
    addFO("jet_offline_puppicand_pt_rel", oc_pt_rel, "offline cand pt / jet pt");
    addFO("jet_offline_puppicand_eta", oc_eta, "offline cand eta in hw units (eta/(pi/720))");
    addFO("jet_offline_puppicand_eta_phys", oc_eta_phys, "offline cand eta");
    addFO("jet_offline_puppicand_deta", oc_deta, "deta in hw units");
    addFO("jet_offline_puppicand_deta_phys", oc_deta_phys, "deta");
    addFO("jet_offline_puppicand_phi", oc_phi, "offline cand phi in hw units");
    addFO("jet_offline_puppicand_phi_phys", oc_phi_phys, "offline cand phi");
    addFO("jet_offline_puppicand_dphi", oc_dphi, "dphi in hw units");
    addFO("jet_offline_puppicand_dphi_phys", oc_dphi_phys, "dphi");
    addFO("jet_offline_puppicand_mass", oc_mass, "offline cand mass");
    addFO("jet_offline_puppicand_dxy", oc_dxy, "int(sqrt(|dxy|)/DXYSQRT_LSB)");
    addFO("jet_offline_puppicand_dxy_phys", oc_dxy_phys, "sqrt(|dxy|)");
    addFO("jet_offline_puppicand_z0", oc_z0, "dz");
    addFO("jet_offline_puppicand_z0_phys", oc_z0_phys, "int(dz/Z0_LSB)");
    addFO("jet_offline_puppicand_quality", oc_quality, "pseudo track quality (else loose bit)");
    addFO("jet_offline_puppicand_puppiweight", oc_puppiweight, "puppi weight");
    addFO("jet_offline_puppicand_isfilled", oc_isfilled, "always 1");
    addFO("jet_offline_puppicand_emid", oc_emid, "always 0 (no EM ID offline)");
    addFO("jet_offline_puppicand_charge", oc_charge, "charge");
    addFO("jet_offline_puppicand_id", oc_id, "pdgId");
    addFO("jet_offline_puppicand_isPhoton", oc_isPhoton, "|pdgId|==22");
    addFO("jet_offline_puppicand_isElectronPlus", oc_isElectronPlus, "pdgId==11");
    addFO("jet_offline_puppicand_isElectronMinus", oc_isElectronMinus, "pdgId==-11");
    addFO("jet_offline_puppicand_isMuonPlus", oc_isMuonPlus, "pdgId==13");
    addFO("jet_offline_puppicand_isMuonMinus", oc_isMuonMinus, "pdgId==-13");
    addFO("jet_offline_puppicand_isHadronPlus", oc_isHadronPlus, "charged hadron, q>0");
    addFO("jet_offline_puppicand_isHadronMinus", oc_isHadronMinus, "charged hadron, q<0");
    addFO("jet_offline_puppicand_isNeutralHadron", oc_isNeutralHadron, "neutral non-photon");
    iEvent.put(std::move(offTab), "L1JetTrainOffCand");
  }

  auto genHTab = std::make_unique<nanoaod::FlatTable>(geninfo.genH_pt.size(), "L1GenH", false);
  genHTab->addColumn<float>("genH_pt", geninfo.genH_pt, "last-copy Higgs pt (== JetNTuplizer genH_pt)");
  genHTab->addColumn<float>("genH_mass", geninfo.genH_mass, "last-copy Higgs mass (== JetNTuplizer genH_mass)");
  iEvent.put(std::move(genHTab), "L1GenH");
  // ngenH == nL1GenH counter branch

  auto genEvtTab = std::make_unique<nanoaod::FlatTable>(1, "L1GenEvent", true);
  genEvtTab->addColumnValue<float>("genHH_pt", geninfo.genHH_pt, "di-Higgs system pt");
  genEvtTab->addColumnValue<float>("genHH_mass", geninfo.genHH_mass, "di-Higgs system mass");
  genEvtTab->addColumnValue<float>("genHT", geninfo.genHT, "sum pt of gen jets pt>30, |eta|<2.4");
  genEvtTab->addColumnValue<float>("genHTall", geninfo.genHTall, "sum pt of all gen jets");
  iEvent.put(std::move(genEvtTab), "L1GenEvent");
}

void L1JetTrainingTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genJets", edm::InputTag("ak4GenJetsNoNu"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("scPuppiJets", edm::InputTag("l1tSC4NGJetTrainProducer", "l1tSC4NGJets"));
  desc.add<edm::InputTag>("scPuppiJetsCorr", edm::InputTag("l1tSC4PFL1PuppiCorrectedEmulator"));
  desc.add<edm::InputTag>("nnTaus", edm::InputTag("l1tNNTauProducerPuppi", "L1PFTausNN"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("l1tLayer2EG", "L1CtTkElectron"));
  desc.add<edm::InputTag>("muons", edm::InputTag("l1tSAMuonsGmt", "prompt"));
  desc.add<edm::InputTag>("genJetsFlavour", edm::InputTag("genFlavourInfo"));
  desc.add<edm::InputTag>("vtx", edm::InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"));
  desc.add<edm::InputTag>("bjetIDs", edm::InputTag("l1tBJetProducerTrain", "L1PFBJets"));
  desc.add<edm::InputTag>("offlineJets", edm::InputTag("slimmedJetsPuppi"));
  desc.add<edm::InputTag>("offlinePVs", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<bool>("doOfflineInfo", true);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(L1JetTrainingTableProducer);
