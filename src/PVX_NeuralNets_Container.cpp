#include <PVX_NeuralNetsCPU.h>

namespace PVX::DeepNeuralNets {
	std::vector<std::pair<float*, size_t>> NeuralNetContainer::MakeDNA() {
		std::vector<std::pair<float*, size_t>> ret;
		for (auto l :Output->Gather()) {
			if (auto d = dynamic_cast<NeuronLayer*>(l); d) {
				auto w = d->GetWeights();
				ret.emplace_back(w.data(), w.size());
			}
		};
		return ret;
	}
	NeuralNetContainer::NeuralNetContainer(OutputLayer* OutLayer) : Output{ OutLayer } {
		auto all = OutLayer->Gather();
		for (auto l : all) {
			auto Inp = dynamic_cast<InputLayer*>(l);
			if (Inp) Inputs.push_back(Inp);
		}
	}
	NeuralNetContainer::NeuralNetContainer(const std::wstring& Filename) {
		PVX::BinLoader bin(Filename.c_str(), "NWRK");
		bin.Process("LYRS", [&](PVX::BinLoader& bin2) {
			bin2.Process("ACTV", [&](PVX::BinLoader& bin3) {
				this->Layers.push_back(ActivationLayer::Load2(bin3));
			});
			bin2.Process("INPT", [&](PVX::BinLoader& bin3) {
				this->Layers.push_back(new InputLayer(bin3));
			});
			bin2.Process("DENS", [&](PVX::BinLoader& bin3) {
				this->Layers.push_back(NeuronLayer::Load2(bin3));
			});
			bin2.Process("ADDR", [&](PVX::BinLoader& bin3) {
				this->Layers.push_back(NeuronAdder::Load2(bin3));
			});
		});
		bin.Process("OUTP", [&](PVX::BinLoader& bin2) {
			int tp, last;
			bin2.Read("TYPE", tp);
			bin2.Read("LAST", last);
			bin2.Execute();
			Output = new OutputLayer(Layers.at(size_t(last)-1), OutputType(tp));
		});
		bin.Execute();
		for (auto l:Layers) {
			l->FixInputs(Layers);
			auto in = dynamic_cast<InputLayer*>(l);
			if (in)Inputs.push_back(in);
		}
	}
	NeuralNetContainer::NeuralNetContainer(const NeuralNetContainer& net) {
		auto layers = net.Output->Gather();
		std::map<NeuralLayer_Base*, size_t> IndexOf;
		size_t i = 1;
		for (auto l : layers) IndexOf[l] = i++;
		for (auto l : layers) Layers.push_back(l->newCopy(IndexOf));
		for (auto l : Layers) l->FixInputs(Layers);
		Output = new OutputLayer(Layers[IndexOf.at(net.Output->LastLayer)], net.Output->Type);
	}
	NeuralNetContainer::~NeuralNetContainer() {
		if (Layers.size()) {
			delete Output;
			for (auto l: Layers) delete l;
		}
	}
	void NeuralNetContainer::Save(const std::wstring& Filename) {
		std::map<NeuralLayer_Base*, size_t> g;
		std::vector<NeuralLayer_Base*> all;
		{
			size_t i = 1;
			for (auto l : Output->Gather()) {
				all.push_back(l);
				g[l] = i++;
			}
		}
		PVX::BinSaver bin(Filename.c_str(), "NWRK");
		bin.Begin("LYRS");
		{
			for (auto l:all) {
				l->Save(bin, g);
			}
		}
		bin.End();
		bin.Begin("OUTP");
		{
			Output->Save(bin, g);
		} 
		bin.End();
	}
	void NeuralNetContainer::SaveCheckpoint() {
		Output->SaveCheckpoint();
	}
	float NeuralNetContainer::LoadCheckpoint() { 
		return Output->LoadCheckpoint();
	}
	void NeuralNetContainer::ResetMomentum() {
		Output->ResetMomentum();
	}
	netData NeuralNetContainer::MakeRawInput(const netData& inp) {
		return Inputs[0]->MakeRawInput(inp);
	}
	netData NeuralNetContainer::MakeRawInput(const std::vector<float>& inp) {
		return Inputs[0]->MakeRawInput(inp);
	}
	std::vector<netData> NeuralNetContainer::MakeRawInput(const std::vector<netData>& inp) {
		std::vector<netData> ret;
		size_t i = 0;
		for (auto l: Inputs)
			ret.push_back(l->MakeRawInput(inp[i++]));
		return ret;
	}

	netData NeuralNetContainer::FromVector(const std::vector<float>& Data) {
		auto r = Output->nOutput();
		netData ret(r, Data.size()/r);
		memcpy(ret.data(), Data.data(), Data.size() * sizeof(float));
		return ret;
	}

	std::vector<float> NeuralNetContainer::ProcessVec(const std::vector<float>& Inp) {
		auto tmp = ProcessRaw(Inputs[0]->MakeRawInput(Inp));
		std::vector<float> ret(tmp.size());
		memcpy(ret.data(), tmp.data(), ret.size() * sizeof(float));
		return ret;
	}

	netData NeuralNetContainer::Process(const netData& inp) {
		Inputs[0]->Input(inp);
		return Output->Result();
	}
	netData NeuralNetContainer::Process(const std::vector<netData>& inp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		return Output->Result();
	}
	netData NeuralNetContainer::ProcessRaw(const netData& inp) {
		Inputs[0]->InputRaw(inp);
		return Output->Result();
	}
	netData NeuralNetContainer::ProcessRaw(const std::vector<netData>& inp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		return Output->Result();
	}
	float NeuralNetContainer::Train(const netData& inp, const netData& outp) {
		Inputs[0]->Input(inp);
		Output->FeedForward();
		return Output->Train(outp);
	}
	float NeuralNetContainer::TrainRaw(const netData& inp, const netData& outp) {
		Inputs[0]->InputRaw(inp);
		Output->FeedForward();
		return Output->Train(outp);
	}
	float NeuralNetContainer::Train(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		Output->FeedForward();
		return Output->Train(outp);
	}
	float NeuralNetContainer::TrainRaw(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		Output->FeedForward();
		return Output->Train(outp);
	}

	float NeuralNetContainer::Error(const netData& inp, const netData& outp) {
		Inputs[0]->Input(inp);
		Output->FeedForward();
		return Output->GetError(outp);
	}
	float NeuralNetContainer::ErrorRaw(const netData& inp, const netData& outp) {
		Inputs[0]->InputRaw(inp);
		Output->FeedForward();
		return Output->GetError(outp);
	}
	float NeuralNetContainer::Error(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		Output->FeedForward();
		return Output->GetError(outp);
	}
	float NeuralNetContainer::ErrorRaw(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		Output->FeedForward();
		return Output->GetError(outp);
	}
}