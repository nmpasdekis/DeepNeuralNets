#include <PVX_NeuralNetsCPU.h>
#include <iostream>
#include <future>

namespace PVX {
	namespace DeepNeuralNets {
		void NeuronAdder::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const {
			bin.Begin("ADDR");
			{
				bin.Write("NDID", int(Id));
				bin.Write("INPC", int(nInput()));
				bin.Begin("LYRS"); {
					for (auto& i: InputLayers)
						bin.write(int(IndexOf.at(i)));
				} bin.End();
			}
			bin.End();
		}
		NeuronAdder* NeuronAdder::Load2(PVX::BinLoader& bin) {
			int inp, Id = -1;
			std::vector<int> layers;
			bin.Read("INPC", inp);
			bin.Read("LYRS", layers);
			bin.Process("NDID", [&](PVX::BinLoader& bin2) { Id = bin2.read<int>(); });
			bin.Execute();
			auto add = new NeuronAdder(inp);
			if (Id>=0)add->Id = Id;
			for (auto l : layers) {
				add->InputLayers.push_back(reinterpret_cast<NeuralLayer_Base*>(((char*)0) + l));
			}
			return add;
		}
		NeuralLayer_Base* NeuronAdder::newCopy(const std::map<NeuralLayer_Base*,size_t>& IndexOf) {
			auto ret =  new NeuronAdder(nInput());
			for (auto l : InputLayers)
				ret->InputLayers.push_back(reinterpret_cast<NeuralLayer_Base*>(IndexOf.at(l)));
			ret->Id = Id;
			return ret;
		}
		NeuronAdder::NeuronAdder(const size_t InputSize) {
			output = netData::Zero(InputSize + 1, 1);
			Id = ++NextId;
		}
		NeuronAdder::NeuronAdder(const std::vector<NeuralLayer_Base*>& Inputs) : NeuronAdder(Inputs[0]->nOutput()) {
			InputLayers = Inputs;
		}
		NeuronAdder::NeuronAdder(const std::string& Name, const size_t InputSize) {
			name = Name;
			output = netData::Zero(InputSize + 1, 1);
			Id = ++NextId;
		}
		NeuronAdder::NeuronAdder(const std::string& Name, const std::vector<NeuralLayer_Base*>& Inputs) : NeuronAdder(Inputs[0]->nOutput()) {
			name = Name;
			InputLayers = Inputs;
		}

		void NeuronAdder::DNA(std::map<void*, WeightData>& Weights) {
			for (auto l : InputLayers)
				l->DNA(Weights);
		}
		void NeuronAdder::FeedForward(int Version) {
			if (Version > FeedVersion) {
				InputLayers[0]->FeedForward(Version);
				output = InputLayers[0]->Output();
				for (auto i = 1; i < InputLayers.size(); i++) {
					InputLayers[i]->FeedForward(Version);
					output += InputLayers[i]->Output();
				}
				output.row(output.rows() - 1) = netData::Ones(1, output.cols());
				FeedVersion = Version;
			}
		}
		void NeuronAdder::BackPropagate(const netData & Gradient) {
			for (auto i : InputLayers) i->BackPropagate(Gradient);
		}
		void NeuronAdder::UpdateWeights() {
			for (auto i: InputLayers) i->UpdateWeights();
		}
		size_t NeuronAdder::nInput() const {
			return output.rows() - 1;
		}

		void NeuronAdder::SetLearnRate(float a) {
			for (auto l : InputLayers)
				l->SetLearnRate(a);
		}

		void NeuronAdder::ResetMomentum() {
			for (auto i : InputLayers)
				i->ResetMomentum();
		}

		void NeuronCombiner::SetLearnRate(float a) {
			for (auto l : InputLayers)
				l->SetLearnRate(a);
		}

		void NeuronCombiner::ResetMomentum() {
			for (auto i : InputLayers)
				i->ResetMomentum();
		}

		void NeuronMultiplier::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const {
			bin.Begin("MULP");
			{
				bin.Write("NDID", int(Id));
				bin.Write("INPC", int(nInput()));
				bin.Begin("LYRS");
				{
					for (auto& i: InputLayers)
						bin.write(int(IndexOf.at(i)));
				} bin.End();
			}
			bin.End();
		}

		NeuronMultiplier* NeuronMultiplier::Load2(PVX::BinLoader& bin) {
			int inp, Id = -1;
			std::vector<int> layers;
			bin.Read("INPC", inp);
			bin.Read("LYRS", layers);
			bin.Process("NDID", [&](PVX::BinLoader& bin2) { Id = bin2.read<int>(); });
			bin.Execute();
			auto add = new NeuronMultiplier(inp);
			if (Id>=0)add->Id = Id;
			for (auto l : layers) {
				add->InputLayers.push_back(reinterpret_cast<NeuralLayer_Base*>(((char*)0) + l));
			}
			return add;
		}

		NeuralLayer_Base* NeuronMultiplier::newCopy(const std::map<NeuralLayer_Base*,size_t>& IndexOf) {
			auto ret = new NeuronMultiplier(nInput());
			for (auto l : InputLayers)
				ret->InputLayers.push_back(reinterpret_cast<NeuralLayer_Base*>(IndexOf.at(l)));
			ret->Id = Id;
			return ret;
		}

		NeuronMultiplier::NeuronMultiplier(const size_t inputs) {
			output = netData::Zero(inputs + 1, 1);
			Id = ++NextId;
		}

		NeuronMultiplier::NeuronMultiplier(const std::vector<NeuralLayer_Base*> & inputs) {
			for (auto& i : inputs) InputLayers.push_back(i);
			output = netData::Zero(InputLayers[0]->Output().rows(), 1);
		}
		void NeuronMultiplier::DNA(std::map<void*, WeightData>& Weights) {
			for (auto l : InputLayers)
				l->DNA(Weights);
		}
		void NeuronMultiplier::FeedForward(int Version) {
			if (Version > FeedVersion) {
				InputLayers[0]->FeedForward(Version);
				auto tmp = InputLayers[0]->Output().array();
				for (auto i = 1; i < InputLayers.size(); i++) {
					InputLayers[i]->FeedForward(Version);
					tmp *= InputLayers[i]->Output().array();
				}
				output = tmp;
				FeedVersion = Version;
			}
		}
		void NeuronMultiplier::BackPropagate(const netData & Gradient) {
			{
				auto tmp = InputLayers[1]->RealOutput().array();
				for (auto i = 2; i < InputLayers.size(); i++) {
					tmp *= InputLayers[i]->RealOutput().array();
				}
				InputLayers[0]->BackPropagate(Gradient.array() * tmp);
			}
			for (int j = 1; j < InputLayers.size(); j++) {
				auto tmp = InputLayers[0]->RealOutput().array();
				for (int i = 1; i < InputLayers.size(); i++)
					if (i != j) 
						tmp *= InputLayers[i]->RealOutput().array();
				InputLayers[j]->BackPropagate(Gradient.array() * tmp);
			}
		}
		void NeuronMultiplier::UpdateWeights() {
			for (auto i: InputLayers) i->UpdateWeights();
		}
		size_t NeuronMultiplier::nInput() const {
			return output.cols();
		}

		void NeuronMultiplier::SetLearnRate(float a) {
			for (auto i : InputLayers)
				i->SetLearnRate(a);
		}
		void NeuronMultiplier::ResetMomentum() {
			for (auto i : InputLayers)
				i->ResetMomentum();
		}

		netData Concat(const std::vector<netData>& M) {
			size_t cols = 0;
			for (auto& m : M) cols += m.cols();
			netData ret(M[0].rows(), cols);
			size_t offset = 0;
			for (auto& m : M) {
				memcpy(ret.data() + offset, m.data(), m.size() * sizeof(float));
				offset += m.size();
			}
			return ret;
		}

		ResNetUtility::ResNetUtility(size_t nInput, size_t nOutput, LayerActivation Activate, TrainScheme Train) :
			First{ nInput, nOutput, Activate, Train },
			Middle{ &First, nOutput, Activate, Train },
			Last{ &Middle, nOutput, LayerActivation::Linear, Train },
			Adder({ &First, &Last }),
			Activation(&Adder, Activate)
		{}

		ResNetUtility::ResNetUtility(NeuralLayer_Base* inp, size_t nOutput, LayerActivation Activate, TrainScheme Train) :
			First{ inp, nOutput, Activate, Train },
			Middle{ &First, nOutput, Activate, Train },
			Last{ &Middle, nOutput, LayerActivation::Linear, Train },
			Adder({ &First, &Last }),
			Activation(&Adder, Activate) {}
		ResNetUtility::ResNetUtility(const std::string& Name, size_t nInput, size_t nOutput, LayerActivation Activate, TrainScheme Train) :
			First { Name + "_First", nInput, nOutput, Activate, Train },
			Middle{ Name + "_Middle",&First, nOutput, Activate, Train },
			Last  { Name + "_Last",  &Middle, nOutput, LayerActivation::Linear, Train },
			Adder(Name + "_Adder", { &First, &Last }),
			Activation(Name + "_Activation", &Adder, Activate) {}
		ResNetUtility::ResNetUtility(const std::string& Name, NeuralLayer_Base* inp, size_t nOutput, LayerActivation Activate, TrainScheme Train) :
			First { Name + "_First", inp, nOutput, Activate, Train },
			Middle{ Name + "_Middle",&First, nOutput, Activate, Train },
			Last  { Name + "_Last",  &Middle, nOutput, LayerActivation::Linear, Train },
			Adder(Name + "_Adder", { &First, &Last }),
			Activation(Name + "_Activation", &Adder, Activate) {}
		ActivationLayer& ResNetUtility::OutputLayer() {
			return Activation;
		}
	}
}