#include <PVX_NeuralNetsCPU.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX::DeepNeuralNets {
	NeuralLayer_Base* RecurrentLayer::newCopy(const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
		auto ret = new RecurrentLayer(
			reinterpret_cast<NeuralLayer_Base*>(IndexOf.at(PreviousLayer)), 
			reinterpret_cast<InputLayer*>(IndexOf.at(RecurrentInput))
		);
		ret->Id = Id;
		return ret;
	}
	RecurrentLayer::RecurrentLayer(NeuralLayer_Base* Input, InputLayer* RecurrentInput):
		RecurrentInput{ RecurrentInput }
	{ 
		PreviousLayer = Input; 
		output = Input->Output();
	}

	void RecurrentLayer::DNA(std::map<void*, WeightData>& Weights) {
		PreviousLayer->DNA(Weights);
	}
	void RecurrentLayer::FeedForward(int Version) {
		if (Version > FeedVersion) {
			PreviousLayer->FeedForward(Version);
			output = PreviousLayer->Output();
			RecurrentInput->InputRaw(output);
			FeedVersion = Version;
		}
	}
	void RecurrentLayer::BackPropagate(const netData& Gradient) {
		PreviousLayer->BackPropagate(Gradient);
	}
	size_t RecurrentLayer::nInput() const {
		return RecurrentInput->nInput();
	}
	void RecurrentLayer::UpdateWeights() {
		PreviousLayer->UpdateWeights();
	}
	void RecurrentLayer::SetLearnRate(float a) {
		PreviousLayer->SetLearnRate(a);
	}
	void RecurrentLayer::ResetMomentum() {
		PreviousLayer->ResetMomentum();
	}

	void RecurrentLayer::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const {
		bin.Begin("RCNN");
		{
			bin.Write("NDID", int(Id));
			bin.Begin("LYRS"); {
				bin.write(int(IndexOf.at(PreviousLayer)));
				bin.write(int(IndexOf.at(RecurrentInput)));
			} bin.End();
		}
		bin.End();
	}
	RecurrentLayer* RecurrentLayer::Load2(PVX::BinLoader& bin) {
		int Id = -1;
		std::vector<int> layers;
		bin.Read("LYRS", layers);
		bin.Process("NDID", [&](PVX::BinLoader& bin2) { Id = bin2.read<int>(); });
		bin.Execute();
		auto rnn = new RecurrentLayer(
			reinterpret_cast<NeuralLayer_Base*>(((char*)0) + layers[0]),
			reinterpret_cast<InputLayer*>(((char*)0) + layers[1])
		);
		if (Id>=0)rnn->Id = Id;
		return rnn;
	}
}