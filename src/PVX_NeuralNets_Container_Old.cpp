#include <PVX_NeuralNetsCPU_Depricated.h>

namespace PVX::DeepNeuralNets {
	std::vector<std::pair<float*, size_t>> NeuralNetContainer_Old::MakeDNA() {
		std::vector<std::pair<float*, size_t>> ret;
		for (auto l :Output->Gather()) {
			if (auto d = dynamic_cast<NeuronLayer*>(l); d) {
				auto w = d->GetWeights();
				ret.emplace_back(w.data(), w.size());
			}
		};
		return ret;
	}
	NeuralNetContainer_Old::NeuralNetContainer_Old(NeuralNetOutput_Base* OutLayer) : Output{ OutLayer } {
		auto all = OutLayer->Gather();
		for (auto l : all) {
			auto Inp = dynamic_cast<InputLayer*>(l);
			if (Inp) Inputs.push_back(Inp);
		}
	}
	NeuralNetContainer_Old::NeuralNetContainer_Old(const std::wstring& Filename) {
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
		});
		bin.Process("OUTP", [&](PVX::BinLoader& bin2) {
			bin2.Process("MSQR", [&](PVX::BinLoader& bin3) {
				Output = new MeanSquareOutput(bin3, Layers);
			});
			bin2.Process("SFTM", [&](PVX::BinLoader& bin3) {
				Output = new SoftmaxOutput(bin3, Layers);
			});
			bin2.Process("SSFM", [&](PVX::BinLoader& bin3) {
				Output = new StableSoftmaxOutput(bin3, Layers);
			});
		});
		bin.Execute();
		for (auto l:Layers) {
			l->FixInputs(Layers);
			auto in = dynamic_cast<InputLayer*>(l);
			if (in)Inputs.push_back(in);
		}
	}
	NeuralNetContainer_Old::~NeuralNetContainer_Old() {
		if (Layers.size()) {
			delete Output;
			for (auto l: Layers) delete l;
		}
	}
	void NeuralNetContainer_Old::Save(const std::wstring& Filename) {
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
	void NeuralNetContainer_Old::SaveCheckpoint() {
		Output->SaveCheckpoint();
	}
	float NeuralNetContainer_Old::LoadCheckpoint() { 
		return Output->LoadCheckpoint();
	}
	void NeuralNetContainer_Old::ResetMomentum() {
		Output->ResetMomentum();
	}
	netData NeuralNetContainer_Old::MakeRawInput(const netData& inp) {
		return Inputs[0]->MakeRawInput(inp);
	}
	netData NeuralNetContainer_Old::MakeRawInput(const std::vector<float>& inp) {
		return Inputs[0]->MakeRawInput(inp);
	}
	std::vector<netData> NeuralNetContainer_Old::MakeRawInput(const std::vector<netData>& inp) {
		std::vector<netData> ret;
		size_t i = 0;
		for (auto l: Inputs)
			ret.push_back(l->MakeRawInput(inp[i++]));
		return ret;
	}

	netData NeuralNetContainer_Old::FromVector(const std::vector<float>& Data) {
		auto r = Output->nOutput();
		netData ret(r, Data.size()/r);
		memcpy(ret.data(), Data.data(), Data.size() * sizeof(float));
		return ret;
	}

	std::vector<float> NeuralNetContainer_Old::ProcessVec(const std::vector<float>& Inp) {
		auto tmp = ProcessRaw(Inputs[0]->MakeRawInput(Inp));
		std::vector<float> ret(tmp.size());
		memcpy(ret.data(), tmp.data(), ret.size() * sizeof(float));
		return ret;
	}

	netData NeuralNetContainer_Old::Process(const netData& inp) {
		Inputs[0]->Input(inp);
		return Output->Result();
	}
	netData NeuralNetContainer_Old::Process(const std::vector<netData>& inp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		return Output->Result();
	}
	netData NeuralNetContainer_Old::ProcessRaw(const netData& inp) {
		Inputs[0]->InputRaw(inp);
		return Output->Result();
	}
	netData NeuralNetContainer_Old::ProcessRaw(const std::vector<netData>& inp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		return Output->Result();
	}
	float NeuralNetContainer_Old::Train(const netData& inp, const netData& outp) {
		Inputs[0]->Input(inp);
		Output->FeedForward();
		return Output->Train(outp);
	}
	float NeuralNetContainer_Old::TrainRaw(const netData& inp, const netData& outp) {
		Inputs[0]->InputRaw(inp);
		Output->FeedForward();
		return Output->Train(outp);
	}
	float NeuralNetContainer_Old::Train(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		Output->FeedForward();
		return Output->Train(outp);
	}
	float NeuralNetContainer_Old::TrainRaw(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		Output->FeedForward();
		return Output->Train(outp);
	}

	float NeuralNetContainer_Old::Error(const netData& inp, const netData& outp) {
		Inputs[0]->Input(inp);
		Output->FeedForward();
		return Output->GetError(outp);
	}
	float NeuralNetContainer_Old::ErrorRaw(const netData& inp, const netData& outp) {
		Inputs[0]->InputRaw(inp);
		Output->FeedForward();
		return Output->GetError(outp);
	}
	float NeuralNetContainer_Old::Error(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		Output->FeedForward();
		return Output->GetError(outp);
	}
	float NeuralNetContainer_Old::ErrorRaw(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		Output->FeedForward();
		return Output->GetError(outp);
	}
}