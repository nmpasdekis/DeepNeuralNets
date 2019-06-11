#include <PVX_NeuralNetsCPU.h>
#include <iostream>
#include "PVX_NeuralNets_Util.inl"

namespace PVX::DeepNeuralNets {
	extern int UseDropout;
	float NeuralLayer_Base::__LearnRate = 0.0001f;
	float NeuralLayer_Base::__Momentum = 0.999f;
	float NeuralLayer_Base::__iMomentum = 0.001f;
	float NeuralLayer_Base::__RMSprop = 0.999f;
	float NeuralLayer_Base::__iRMSprop = 0.1f;
	float NeuralLayer_Base::__Dropout = 0.8f;
	float NeuralLayer_Base::__iDropout = 1.0f / 0.8f;
	float NeuralLayer_Base::__L2 = 0.0f;
	int NeuralLayer_Base::OverrideOnLoad = 0;

	netData myRandom(int r, int c, float Max) {
		return Max * netData::Random(r, c);
	}

	//int Check(netData mat) {
	//	float* dt = mat.data();
	//	size_t sz = mat.cols() * mat.rows();
	//	for (auto i = 0; i < sz; i++) {
	//		if (dt[i] == dt[i] + 1.0f) {
	//			exit(-1);
	//			return 0;
	//		}
	//	}
	//	return 1;
	//}

	netData NeuralLayer_Base::Output() {
		return output;
	}
	netData NeuralLayer_Base::RealOutput() {
		return outPart(output);
	}
	float NeuralLayer_Base::LearnRate() {
		return __LearnRate;
	}
	void NeuralLayer_Base::LearnRate(float Alpha) {
		__LearnRate = Alpha;
	}
	float NeuralLayer_Base::Momentum() {
		return __Momentum;
	}
	void NeuralLayer_Base::Momentum(float Beta) {
		__Momentum = Beta;
		__iMomentum = 1.0f - Beta;
	}
	float NeuralLayer_Base::RMSprop() {
		return __RMSprop;
	}
	void NeuralLayer_Base::RMSprop(float Beta) {
		__RMSprop = Beta;
		__iRMSprop = 1.0f - Beta;
	}
	void NeuralLayer_Base::L2Regularization(float lambda) {
		__L2 = lambda;
	}
	float NeuralLayer_Base::L2Regularization() {
		return __L2;
	}
	float NeuralLayer_Base::Dropout() {
		return __Dropout;
	}
	void NeuralLayer_Base::Dropout(float Beta) {
		__Dropout = Beta;
		__iDropout = 1.0f / Beta;
	}
	int NeuralLayer_Base::nOutput() {
		return output.rows() - 1;
	}

	int NeuralLayer_Base::BatchSize() {
		return output.cols();
	}

	void NeuralLayer_Base::Gather(std::set<NeuralLayer_Base*>& g) {
		g.insert(this);
		if (PreviousLayer)PreviousLayer->Gather(g);
		if (InputLayers.size())for (auto& i:InputLayers)i->Gather(g);
	}

	void NeuralLayer_Base::FixInputs(const std::vector<NeuralLayer_Base*>& ids) {
		if (PreviousLayer) PreviousLayer = ids[(*(int*)&PreviousLayer)-1ll];
		else for (auto& l : InputLayers)
			l = ids[(*(int*)&l)-1ll];
	}

	void NeuralLayer_Base::Input(NeuralLayer_Base* inp) {
		PreviousLayer = inp;
	}

	void NeuralLayer_Base::Inputs(const std::vector<NeuralLayer_Base*>& inp) {
		InputLayers = inp;
	}

	std::set<NeuralLayer_Base*> NeuralNetOutput_Base::Gather() {
		std::set<NeuralLayer_Base*> g;
		LastLayer->Gather(g);
		return g;
	}

	void NeuralNetOutput_Base::FeedForward() {
		LastLayer->FeedForward(++Version);
		auto tmp = LastLayer->Output();
		output = outPart(tmp);
	}

	NeuralNetOutput_Base::NeuralNetOutput_Base(NeuralLayer_Base* Last) : LastLayer{ Last }, output{ 1, Last->Output().cols() } { }
	void NeuralNetOutput_Base::Result(float* res) {
		auto r = Result();
		memcpy(res, r.data(), sizeof(float) * r.cols());
	}

	const netData& NeuralNetOutput_Base::Result() {
		FeedForward();
		return output;
	}

	int NeuralNetOutput_Base::nOutput() {
		return LastLayer->output.rows() - 1;
	}

	void NeuralNetOutput_Base::SaveCheckpoint() {
		if (Checkpoint.Layers.size()==0) {
			Checkpoint = GetDNA();
		}
		CheckpointDNA = Checkpoint.GetData();
		CheckpointError = Error;
	}

	float NeuralNetOutput_Base::LoadCheckpoint() {
		Error = CheckpointError;
		Checkpoint.SetData(CheckpointDNA.data());
		return Error;
	}

	void NeuralNetOutput_Base::ResetMomentum() {
		LastLayer->ResetMomentum();
	}

	void NeuronLayer::DNA(std::map<void*, WeightData>& w) {
		if (!w.count(this)) {
			WeightData ret;
			ret.Weights = Weights.data();
			ret.Count = Weights.size();
			w[this] = ret;
		}
		PreviousLayer->DNA(w);
	}

	NetDNA NeuralNetOutput_Base::GetDNA() {
		std::map<void*, WeightData> data;
		LastLayer->DNA(data);
		NetDNA ret;
		ret.Size = 0;
		for (auto& [l, dt] : data) {
			dt.Offset = ret.Size;
			ret.Size += dt.Count;
			ret.Layers.push_back(dt);
		}
		return ret;
	}

	std::vector<float> NetDNA::GetData() {
		std::vector<float> ret(Size);
		for (auto& w : Layers) {
			memcpy_s(ret.data() + w.Offset, sizeof(float) * w.Count, w.Weights, sizeof(float) * w.Count);
		}
		return ret;
	}
	void NetDNA::SetData(const float* Data) {
		for (auto& w : Layers)
			memcpy(w.Weights, &Data[w.Offset], sizeof(float) * w.Count);
	}

	//void NeuralNetOutput_Base::Save(const wchar_t* Filename) {

	//}
	//void NeuralNetOutput_Base::Load(const wchar_t* Filename) {

	//}

}