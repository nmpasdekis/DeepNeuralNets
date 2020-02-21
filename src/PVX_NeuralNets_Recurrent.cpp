#include <PVX_NeuralNetsCPU.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX::DeepNeuralNets {
	NeuralLayer_Base* RecurrentLayer::newCopy(const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
		auto ret = new RecurrentLayer(
			reinterpret_cast<NeuralLayer_Base*>(IndexOf.at(PreviousLayer)), 
			reinterpret_cast<RecurrentInput*>(IndexOf.at(RNN_Input))
		);
		ret->Id = Id;
		return ret;
	}
	RecurrentLayer::RecurrentLayer(NeuralLayer_Base* Input, RecurrentInput* RecurrentInput):
		RNN_Input{ RecurrentInput }
	{ 
		PreviousLayer = Input; 
		output = Input->Output();
	}
	void RecurrentLayer::FeedForward(int Version) {
		if (Version > FeedVersion) {
			RNN_Input->FeedForward(Version);
			int bSize = RNN_Input->BatchSize();
			RNN_Input->SetFeedVersion(Version + bSize);
			if (output.cols() != bSize) {
				output = netData::Zero(output.rows(), bSize);
				output(output.rows()-1, 0) = 1.0f;
			}
			for (int i = 0; i<bSize; i++) {
				RNN_Input->output.block(0, 0, RNN_Input->RecurrentNeuronCount, 1) = output.block(0, i, output.rows() - 1, 1);
				RNN_Input->FeedIndex(i);
				PreviousLayer->FeedForward(Version + i);
				output.col((i + 1) % bSize) = PreviousLayer->Output();
			}
			output.row(output.rows()-1) = netData::Ones(1, bSize);
			SetFeedVersion(Version);
		}
	}

	void RecurrentLayer::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const {
		bin.Begin("RNNe");
		{
			bin.Write("NDID", int(Id));
			bin.Begin("LYRS"); {
				bin.write(int(IndexOf.at(PreviousLayer)));
				bin.write(int(IndexOf.at(RNN_Input)));
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
			reinterpret_cast<RecurrentInput*>(((char*)0) + layers[1])
		);
		if (Id>=0)rnn->Id = Id;
		return rnn;
	}

	void RecurrentLayer::DNA(std::map<void*, WeightData>& Weights) {
		PreviousLayer->DNA(Weights);
	}
	void RecurrentLayer::BackPropagate(const netData& Gradient) {
		PreviousLayer->BackPropagate(Gradient);
	}
	size_t RecurrentLayer::nInput() const {
		return PreviousLayer->nInput();
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



	RecurrentInput::RecurrentInput(NeuralLayer_Base* Input, int RecurrentNeurons) : RecurrentNeuronCount{ RecurrentNeurons } {
		PreviousLayer = Input;
		output = netData::Zero(RecurrentNeurons + Input->nOutput() + 1, 1);
		rnnData = netData::Zero(RecurrentNeurons, 1);
	}
	void RecurrentInput::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const {
		bin.Begin("RNNs");
		{
			bin.Write("NDID", int(Id));
			bin.Write("INPC", int(nInput()));
			bin.Write("RNNC", int(RecurrentNeuronCount));
			bin.Begin("LYRS"); {
				bin.write(int(IndexOf.at(PreviousLayer)));
			} bin.End();
		}
		bin.End();
	}
	NeuralLayer_Base* RecurrentInput::newCopy(const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
		auto ret = new RecurrentInput(
			reinterpret_cast<NeuralLayer_Base*>(IndexOf.at(PreviousLayer)),
			RecurrentNeuronCount
		);
		ret->Id = Id;
		return ret;
	}
	RecurrentInput* RecurrentInput::Load2(PVX::BinLoader& bin) {
		int Id = -1;
		int RecurrentNeuronCount;
		std::vector<int> layers;
		bin.Read("LYRS", layers);
		bin.Process("NDID", [&](PVX::BinLoader& bin2) { Id = bin2.read<int>(); });
		bin.Read("RNNC", RecurrentNeuronCount);
		bin.Execute();
		auto rnn = new RecurrentInput(
			reinterpret_cast<NeuralLayer_Base*>(((char*)0) + layers[0]),
			RecurrentNeuronCount
		);
		if (Id>=0)rnn->Id = Id;
		return rnn;
	}
	int RecurrentInput::BatchSize() {
		return int(PreviousLayer->Output().cols());
	}
	void RecurrentInput::FeedIndex(int i) {
		output.block(RecurrentNeuronCount, 0, rnnData.rows(), 1) = rnnData.col(i);
	}
	void RecurrentInput::FeedForward(int ver) {
		PreviousLayer->FeedForward(ver);
		rnnData = PreviousLayer->Output();
	}




	void RecurrentInput::DNA(std::map<void*, WeightData>& Weights) {
		PreviousLayer->DNA(Weights);
	}
	void RecurrentInput::BackPropagate(const netData& Gradient) {
		PreviousLayer->BackPropagate(Gradient);
	}
	size_t RecurrentInput::nInput() const {
		return output.rows() - 1;
	}
	void RecurrentInput::UpdateWeights() {
		PreviousLayer->UpdateWeights();
	}
	void RecurrentInput::SetLearnRate(float a) {
		PreviousLayer->SetLearnRate(a);
	}
	void RecurrentInput::ResetMomentum() {
		PreviousLayer->ResetMomentum();
	}
}