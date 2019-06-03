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
	NeuralNetContainer::NeuralNetContainer(NeuralNetOutput* OutLayer) : Output{ OutLayer } {
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
	Eigen::MatrixXf NeuralNetContainer::MakeRawInput(const Eigen::MatrixXf& inp) {
		return Inputs[0]->MakeRawInput(inp);
	}
	std::vector<Eigen::MatrixXf> NeuralNetContainer::MakeRawInput(const std::vector<Eigen::MatrixXf>& inp) {
		std::vector<Eigen::MatrixXf> ret;
		size_t i = 0;
		for (auto l: Inputs)
			ret.push_back(l->MakeRawInput(inp[i++]));
		return ret;
	}
	Eigen::MatrixXf NeuralNetContainer::Process(const Eigen::MatrixXf& inp) {
		Inputs[0]->Input(inp);
		return Output->Result();
	}
	Eigen::MatrixXf NeuralNetContainer::Process(const std::vector<Eigen::MatrixXf>& inp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		return Output->Result();
	}
	Eigen::MatrixXf NeuralNetContainer::ProcessRaw(const Eigen::MatrixXf& inp) {
		Inputs[0]->InputRaw(inp);
		return Output->Result();
	}
	Eigen::MatrixXf NeuralNetContainer::ProcessRaw(const std::vector<Eigen::MatrixXf>& inp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		return Output->Result();
	}
	float NeuralNetContainer::Train(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp) {
		Inputs[0]->Input(inp);
		return Output->Train(outp);
	}
	float NeuralNetContainer::TrainRaw(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp) {
		Inputs[0]->InputRaw(inp);
		return Output->Train(outp);
	}
	float NeuralNetContainer::Train(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		return Output->Train(outp);
	}
	float NeuralNetContainer::TrainRaw(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		return Output->Train(outp);
	}

	float NeuralNetContainer::Error(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp) {
		Inputs[0]->Input(inp);
		Output->FeedForward();
		return Output->GetError(outp);
	}
	float NeuralNetContainer::ErrorRaw(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp) {
		Inputs[0]->InputRaw(inp);
		Output->FeedForward();
		return Output->GetError(outp);
	}
	float NeuralNetContainer::Error(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		Output->FeedForward();
		return Output->GetError(outp);
	}
	float NeuralNetContainer::ErrorRaw(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		Output->FeedForward();
		return Output->GetError(outp);
	}
}