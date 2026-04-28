#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC4NGJetID.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/inversion.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/log.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <cmath>
#include <codecvt>
#include <iostream>

typedef ap_fixed<32,16,AP_RND,AP_SAT,0> input_t;
typedef ap_fixed<32,16,AP_RND,AP_SAT,0> input12_t;
typedef ap_ufixed<17,1,AP_RND,AP_SAT,0> input29_t;
typedef ap_uint<1> input17_t;

// Then define the struct
struct ModelInputs {
    input_t* basic_input;
    input29_t* constituent_fraction;
    input12_t* jet_features;
    input17_t* pt_mask;
};

L1TSC4NGJetID::L1TSC4NGJetID(const std::shared_ptr<hls4mlEmulator::Model> model, int iNParticles, bool debug)
    : modelRef_(model)
 {
  NNvectorVar_.clear();
  fNParticles_ = iNParticles;
  isDebugEnabled_ = debug;

  fPt_ = std::make_unique<inputtype[]>(fNParticles_);
  fPt_rel_ = std::make_unique<inputtype[]>(fNParticles_);
  fEta_ = std::make_unique<inputtype[]>(fNParticles_);
  fDEta_ = std::make_unique<inputtype[]>(fNParticles_);
  fDPhi_ = std::make_unique<inputtype[]>(fNParticles_);
  fPt_log_ = std::make_unique<inputtype[]>(fNParticles_);
  fMass_ = std::make_unique<inputtype[]>(fNParticles_);
  fZ0_ = std::make_unique<inputtype[]>(fNParticles_);
  fDxy_ = std::make_unique<inputtype[]>(fNParticles_);
  fIs_filled_ = std::make_unique<inputtype[]>(fNParticles_);
  fPuppi_weight_ = std::make_unique<inputtype[]>(fNParticles_);
  fEmID_ = std::make_unique<inputtype[]>(fNParticles_);
  fQuality_ = std::make_unique<inputtype[]>(fNParticles_);

  fId_ = std::make_unique<inputtype[]>(fNParticles_);
  fCharge_ = std::make_unique<inputtype[]>(fNParticles_);

   fJetPt_ = 0;
   fJetEta_ = 0;
}

void L1TSC4NGJetID::setNNVectorVar() {
  NNvectorVar_.clear();
  if (isDebugEnabled_) {
    LogDebug("L1TSC4NGJetID") << "\n ===== Input Vector =====" << std::endl;
  }

  for (int i0 = 0; i0 < fNParticles_; i0++) {
    bool filled = fIs_filled_.get()[i0] == 1;
    inputtype null_value = 0;

    NNvectorVar_.push_back(filled ? fPt_.get()[i0] : null_value);      // pt
    NNvectorVar_.push_back(filled ? fPt_rel_.get()[i0] : null_value);  // pT as a fraction of jet pT
    NNvectorVar_.push_back(filled ? fPt_log_.get()[i0] : null_value);  // pt log
    NNvectorVar_.push_back(filled ? fEta_.get()[i0] : null_value);      // eta
    NNvectorVar_.push_back(filled ? fDEta_.get()[i0] : null_value);    // dEta from jet axis
    NNvectorVar_.push_back(filled ? fDPhi_.get()[i0] : null_value);    // dPhi from jet axis
    NNvectorVar_.push_back(filled ? fMass_.get()[i0] : null_value);    // Mass
    NNvectorVar_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::PHOTON) : null_value);    // Photon
    NNvectorVar_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::ELEPLUS) : null_value);   // Positron
    NNvectorVar_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::ELEMINUS) : null_value);  // Electron
    NNvectorVar_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::MUPLUS) : null_value);    // Anti-muon
    NNvectorVar_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::MUMINUS) : null_value);   // Muon
    NNvectorVar_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::HADZERO) : null_value);
    NNvectorVar_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::HADPLUS) : null_value);   // Anti-Pion
    NNvectorVar_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::HADMINUS) : null_value);  // Pion
    NNvectorVar_.push_back(filled ? fZ0_.get()[i0] : null_value);                                           // z0
    NNvectorVar_.push_back(filled ? fDxy_.get()[i0] : null_value);                                          // dxy
    NNvectorVar_.push_back(filled ? fIs_filled_.get()[i0] : null_value);                                    // isfilled
    NNvectorVar_.push_back(filled ? fPuppi_weight_.get()[i0] : null_value);  // puppi weight
    NNvectorVar_.push_back(filled ? fEmID_.get()[i0] : null_value);          // emID
    NNvectorVar_.push_back(filled ? fQuality_.get()[i0] : null_value);       // quality

    if (isDebugEnabled_) {
      LogDebug("L1TSC4NGJetID") << "Particle: " << i0 << "\n"
                                << "pT: " << NNvectorVar_[i0 * 21]
                                << " | "
                                   "pT rel: "
                                << NNvectorVar_[i0 * 21 + 1]
                                << " | "
                                   "pT log: "
                                << NNvectorVar_[i0 * 21 + 2]
                                << " | "
                                   "eta: "
                                << NNvectorVar_[i0 * 21 + 3]
                                << " | "
                                   "dEta: "
                                << NNvectorVar_[i0 * 21 + 4]
                                << " | "
                                   "dPhi: "
                                << NNvectorVar_[i0 * 21 + 5]
                                << " | "
                                   "mass: "
                                << NNvectorVar_[i0 * 21 + 6]
                                << " | "
                                   "photon ID: "
                                << NNvectorVar_[i0 * 21 + 7]
                                << " | "
                                   "electron + ID: "
                                << NNvectorVar_[i0 * 21 + 8]
                                << " | "
                                   "electron - ID: "
                                << NNvectorVar_[i0 * 21 + 9]
                                << " | "
                                   "muon + ID: "
                                << NNvectorVar_[i0 * 21 + 10]
                                << " | "
                                   "muon - ID: "
                                << NNvectorVar_[i0 * 21 + 11]
                                << " | "
                                   "neutral hadron ID: "
                                << NNvectorVar_[i0 * 21 + 12]
                                << " | "
                                   "hadron + ID: "
                                << NNvectorVar_[i0 * 21 + 13]
                                << " | "
                                   "hadron - ID: "
                                << NNvectorVar_[i0 * 21 + 14]
                                << " | "
                                   "z0: "
                                << NNvectorVar_[i0 * 21 + 15]
                                << " | "
                                   "sqrt Dxy: "
                                << NNvectorVar_[i0 * 21 + 16]
                                << " | "
                                   "is filled: "
                                << NNvectorVar_[i0 * 21 + 17]
                                << " | "
                                   "puppi weight: "
                                << NNvectorVar_[i0 * 21 + 18]
                                << " | "
                                   "ElectroMagnetic ID: "
                                << NNvectorVar_[i0 * 21 + 19]
                                << " | "
                                   "Track Quality: "
                                << NNvectorVar_[i0 * 21 + 20] << " | "
                                << "===========" << std::endl;
    }
  }
  // After the particle loop
  NNvectorVar_.push_back(fJetPt_);
  NNvectorVar_.push_back(fJetEta_);
}

L1TSC4NGJetID::outputpairtype L1TSC4NGJetID::EvaluateNNFixed() {
  const int NInputs = 336;
  const int NParticles = fNParticles_;
  classtype classresult;
  regressiontype regressionresult;

  inputtype fillzero = 0.0;

  // Base Input
  input_t baseInput[NInputs] = {};
  std::fill(baseInput, baseInput + NInputs, fillzero);
  for (unsigned int i = 0; i < NInputs; i++) {
      baseInput[i] = NNvectorVar_[i];
  }

   // Mask for pt
   input17_t modelPtMask[NParticles] = {};
   for (int i = 0; i < NParticles; i++) {
      modelPtMask[i] = fIs_filled_.get()[i];
   }

  // Just the constituents pt fractions
   input_t jetPt = NNvectorVar_[NInputs];
   input29_t pTFractions[NParticles] = {};
   for (int i = 0; i < NParticles; i++) {
      pTFractions[i] = fPt_rel_.get()[i];  // pt rel feature as a fraction of jet pt
   }

  // Jet Features
   input12_t modelJetFeatures[2] = {};
   modelJetFeatures[0] = l1ct::log_with_shift<l1ct::pt_t, inputtype, 256>(NNvectorVar_[NInputs]); // log pt
   modelJetFeatures[1] = (NNvectorVar_[NInputs + 1] < 0)
                       ? input12_t(-NNvectorVar_[NInputs + 1])
                       : input12_t(NNvectorVar_[NInputs + 1]); // abs jet eta

   // Fill the model input struct
   ModelInputs inputs;

   inputs.basic_input          = baseInput;
   inputs.constituent_fraction = pTFractions;
   inputs.jet_features         = modelJetFeatures;
   inputs.pt_mask              = modelPtMask;

  pairtype modelResult;

  modelRef_->prepare_input(inputs);
  modelRef_->predict();
  modelRef_->read_result(&modelResult);

  outputpairtype modelResult_forOutput;
  if (isDebugEnabled_) {
    LogDebug("L1TSC4NGJetID") << "\n ===== Jet ID Output Score =====" << std::endl;
  }
  for (unsigned int i = 0; i < 9; i++) {
    // Cast model output to jet tag score datatype
    modelResult_forOutput.second[i] = l1ct::jet_tag_score_t(modelResult.second[i]);
    if (isDebugEnabled_) {
      LogDebug("L1TSC4NGJetID") << l1ct::JetTagClassHandler::tagClassesDefault_[i] << " : " << modelResult.second[i]
                                << " Cast to Jet Class type: " << modelResult_forOutput.second[i] << std::endl;
    }
  }
  // Cast model output to transient regression score for jet pt multiplication
  modelResult_forOutput.first[0] = output_regression_type(modelResult.first[0]);
  if (isDebugEnabled_) {
    LogDebug("L1TSC4NGJetID") << "\n ===== Jet pT Correction Output ===== \n"
                              << modelResult.first[0] << " Cast to Jet pT type: " << modelResult_forOutput.first[0]
                              << std::endl;
  }
  return modelResult_forOutput;
}  //end EvaluateNNFixed

L1TSC4NGJetID::outputpairtype L1TSC4NGJetID::computeFixed(const l1t::PFJet &iJet) {
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    fPt_rel_.get()[i0] = 0;
    fPt_.get()[i0] = 0;
    fEta_.get()[i0] = 0;
    fDEta_.get()[i0] = 0;
    fDPhi_.get()[i0] = 0;
    fPt_log_.get()[i0] = 0;
    fMass_.get()[i0] = 0;
    fZ0_.get()[i0] = 0;
    fDxy_.get()[i0] = 0;
    fIs_filled_.get()[i0] = 0;
    fPuppi_weight_.get()[i0] = 0;
    fEmID_.get()[i0] = 0;
    fQuality_.get()[i0] = 0;

    fId_.get()[i0] = 0;
    fCharge_.get()[i0] = 0;
  }
  auto iParts = iJet.constituents();
  // Use stable sort for deterministic ordering when equal pT to match FW
  std::stable_sort(iParts.begin(), iParts.end(), [](edm::Ptr<l1t::PFCandidate> i, edm::Ptr<l1t::PFCandidate> j) {
    return (i->pt() > j->pt());
  });

  l1ct::Jet ctJet = l1ct::Jet::unpack(iJet.getHWJetCT());
  inputtype jet_pt_ = inputtype(ctJet.hwPt);
  inputtype jet_phi_ = inputtype(ctJet.hwPhi);
  inputtype jet_eta_ = inputtype(ctJet.hwEta);

  // Fill jet level features
  fJetPt_ = jet_pt_;
  fJetEta_ = jet_eta_;
  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 >= (unsigned int)fNParticles_)
      break;
    l1ct::PuppiObj puppicand = l1ct::PuppiObj::unpack(iParts[i0]->encodedPuppi64());
    fPt_.get()[i0] = inputtype(puppicand.hwPt);

    constexpr int INV_LUT_SIZE = 1024;
    inputtype inv_jet_pt = l1ct::invert_with_shift<l1ct::pt_t, inputtype, INV_LUT_SIZE>(jet_pt_);

    fPt_rel_.get()[i0] = inputtype(puppicand.hwPt) * inv_jet_pt;

    inputtype const_eta = inputtype(puppicand.hwEta);
    fEta_.get()[i0] = (const_eta < 0)
      ? inputtype(-const_eta)
      : inputtype(const_eta);

    L1SCJetEmu::detaphi_t dphi(puppicand.hwPhi - jet_phi_);
    // phi wrap
    L1SCJetEmu::detaphi_t dphi0 = dphi > L1SCJetEmu::detaphi_t(l1ct::Scales::INTPHI_PI)
                                      ? L1SCJetEmu::detaphi_t(dphi - l1ct::Scales::INTPHI_TWOPI)
                                      : L1SCJetEmu::detaphi_t(dphi);
    L1SCJetEmu::detaphi_t dphi1 = dphi < L1SCJetEmu::detaphi_t(-l1ct::Scales::INTPHI_PI)
                                      ? L1SCJetEmu::detaphi_t(l1ct::Scales::INTPHI_TWOPI + dphi)
                                      : L1SCJetEmu::detaphi_t(dphi);
    inputtype dphiw = inputtype(dphi > L1SCJetEmu::detaphi_t(0) ? dphi0 : dphi1);

    fDEta_.get()[i0] = jet_eta_ - inputtype(puppicand.hwEta);
    fDPhi_.get()[i0] = dphiw;

    constexpr int LOG_LUT_SIZE = 256;
    inputtype log_pt = l1ct::log_with_shift<l1ct::pt_t, inputtype, LOG_LUT_SIZE>(puppicand.hwPt);
    fPt_log_.get()[i0] = log_pt;

    inputtype massCand = L1TSC4NGJet::candidate_mass<inputtype>(puppicand);
    fMass_.get()[i0] = inputtype(massCand);

    fZ0_.get()[i0] = puppicand.hwId.charged() ? inputtype(puppicand.hwZ0()) : inputtype(0);
    fDxy_.get()[i0] = puppicand.hwId.charged() ? inputtype(puppicand.hwDxy()) : inputtype(0);
    fIs_filled_.get()[i0] = inputtype(1);
    fPuppi_weight_.get()[i0] = puppicand.hwId.neutral() ? inputtype(puppicand.hwPuppiW()) : inputtype(0);
    fEmID_.get()[i0] = puppicand.hwId.neutral() ? inputtype(puppicand.hwEmID()) : inputtype(0);
    fQuality_.get()[i0] = puppicand.hwId.charged() ? inputtype(puppicand.hwTkQuality()) : inputtype(0);

    fCharge_.get()[i0] = inputtype(puppicand.hwId.charged());
    fId_.get()[i0] = inputtype(puppicand.hwId.bits);
  }
  setNNVectorVar();
  return EvaluateNNFixed();
}
