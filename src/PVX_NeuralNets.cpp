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
	netData NeuralLayer_Base::RealOutput(int Index) {
		return outPart(output, Index);
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

	void NeuralLayer_Base::SetLearnRate(float Beta) {
		if (PreviousLayer)PreviousLayer->SetLearnRate(Beta);
		for (auto& p : InputLayers) p->SetLearnRate(Beta);
	}
	void NeuralLayer_Base::SetRMSprop(float Beta) {
		if (PreviousLayer)PreviousLayer->SetRMSprop(Beta);
		for (auto& p : InputLayers) p->SetRMSprop(Beta);
	}
	void NeuralLayer_Base::SetMomentum(float Beta) {
		if (PreviousLayer)PreviousLayer->SetMomentum(Beta);
		for (auto& p : InputLayers) p->SetMomentum(Beta);
	}
	void NeuralLayer_Base::ResetMomentum() {
		if (PreviousLayer)PreviousLayer->ResetMomentum();
		for (auto& p : InputLayers) p->ResetMomentum();
	}

	void NeuralLayer_Base::Gather(std::set<NeuralLayer_Base*>& g) {
		g.insert(this);
		if (PreviousLayer)PreviousLayer->Gather(g);
		if (InputLayers.size())for (auto& i:InputLayers)i->Gather(g);
	}

	void NeuralLayer_Base::FixInputs(const std::vector<NeuralLayer_Base*>& ids) {
		if (PreviousLayer) {
			PreviousLayer = ids[(*(int*)&PreviousLayer)-1ll];
			PreviousLayer->OutputRefCount++;
		}
		else for (auto& l : InputLayers) {
			l = ids[(*(int*)&l)-1ll];
			l->OutputRefCount++;
		}
	}

	void NeuralLayer_Base::Input(NeuralLayer_Base* inp) {
		PreviousLayer = inp;
		inp->OutputRefCount++;
	}

	void NeuralLayer_Base::Inputs(const std::vector<NeuralLayer_Base*>& inp) {
		InputLayers = inp;
		for (auto i: InputLayers)
			i->OutputRefCount++;
	}

	void NetDNA::GetData(std::vector<float>& Data) {
		if (Size!=Data.size())Data.resize(Size);
		for (auto& w : Layers) {
			memcpy_s(Data.data() + w.Offset, sizeof(float) * w.Count, w.Weights, sizeof(float) * w.Count);
		}
	}

	std::vector<float> NetDNA::GetData() {
		std::vector<float> ret(Size);
		for (auto& w : Layers) {
			memcpy_s(ret.data() + w.Offset, sizeof(float) * w.Count, w.Weights, sizeof(float) * w.Count);
		}
		return std::move(ret);
	}
	void NetDNA::SetData(const float* Data) {
		for (auto& w : Layers)
			memcpy(w.Weights, &Data[w.Offset], sizeof(float) * w.Count);
	}
}