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
	size_t NeuralLayer_Base::NextId = 0;

	netData myRandom(int r, int c, float Max) {
		return Max * netData::Random(r, c);
	}

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
	size_t NeuralLayer_Base::nOutput() const {
		return output.rows() - 1;
	}

	size_t NeuralLayer_Base::BatchSize() const {
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

	void NeuronLayer::DNA(std::map<void*, WeightData>& w) {
		if (!w.count(this)) {
			WeightData ret;
			ret.Weights = Weights.data();
			ret.Count = Weights.size();
			w[this] = ret;
		}
		PreviousLayer->DNA(w);
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
}