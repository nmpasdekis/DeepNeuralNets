#include <PVX_NeuralNetsCPU.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX::DeepNeuralNets {
	netData NetContainer::MakeRawInput(const netData& inp) {
		return Inputs[0]->MakeRawInput(inp);
	}
	netData NetContainer::MakeRawInput(const std::vector<float>& inp) {
		return Inputs[0]->MakeRawInput(inp);
	}
	std::vector<netData> NetContainer::MakeRawInput(const std::vector<netData>& inp) {
		std::vector<netData> ret;
		size_t i = 0;
		for (auto l: Inputs)
			ret.push_back(l->MakeRawInput(inp[i++]));
		return ret;
	}

	NetContainer::NetContainer(NeuralLayer_Base* Last, OutputType Type) :LastLayer{ Last }, Type{ Type } {
		switch (Type) {
			case PVX::DeepNeuralNets::OutputType::MeanSquare:
				FeedForward = [this] { FeedForwardMeanSquare(); };
				GetErrorFnc = [this](const netData& Data) { return GetError_MeanSquare(Data); };
				TrainFnc = [this](const netData& Data) { return Train_MeanSquare(Data); };
				break;
			case PVX::DeepNeuralNets::OutputType::SoftMax:
				FeedForward = [this] { FeedForwardSoftMax(); };
				GetErrorFnc = [this](const netData& Data) { return GetError_SoftMax(Data); };
				TrainFnc = [this](const netData& Data) { return Train_SoftMax(Data); };
				break;
			case PVX::DeepNeuralNets::OutputType::StableSoftMax:
				FeedForward = [this] { FeedForwardStableSoftMax(); };
				GetErrorFnc = [this](const netData& Data) { return GetError_SoftMax(Data); };
				TrainFnc = [this](const netData& Data) { return Train_SoftMax(Data); };
				break;
			default:
				break;
		}

		std::set<NeuralLayer_Base*> all;
		LastLayer->Gather(all);
		for (auto l : all) {
			auto Inp = dynamic_cast<InputLayer*>(l);
			if (Inp) 
				Inputs.push_back(Inp);
			else {
				auto dense = dynamic_cast<NeuronLayer*>(l);
				if (dense) {
					DenseLayers.push_back(dense);
				}
			}
		}
		std::sort(DenseLayers.begin(), DenseLayers.end(), [](NeuronLayer* a, NeuronLayer* b) {
			return a->Id<b->Id;
		});
	}
	NetContainer::~NetContainer() {
		if (Layers.size()) {
			for (auto l: Layers) delete l;
		}
	}

	netData NetContainer::Process(const netData& inp) const {
		Inputs[0]->Input(inp);
		LastLayer->FeedForward(++Version);
		return LastLayer->Output();
	}
	netData NetContainer::Process(const std::vector<netData>& inp) const {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		LastLayer->FeedForward(++Version);
		return LastLayer->Output();
	}
	netData NetContainer::ProcessRaw(const netData& inp) const {
		Inputs[0]->InputRaw(inp);
		LastLayer->FeedForward(++Version);
		return LastLayer->Output();
	}
	netData NetContainer::ProcessRaw(const std::vector<netData>& inp) const {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		LastLayer->FeedForward(++Version);
		return LastLayer->Output();
	}
	float NetContainer::Train(const netData& inp, const netData& outp) {
		Inputs[0]->Input(inp);
		LastLayer->FeedForward(++Version);
		return TrainFnc(outp);
	}
	float NetContainer::TrainRaw(const netData& inp, const netData& outp) {
		Inputs[0]->InputRaw(inp);
		LastLayer->FeedForward(++Version);
		return TrainFnc(outp);
	}
	float NetContainer::Train(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		LastLayer->FeedForward(++Version);
		return TrainFnc(outp);
	}
	float NetContainer::TrainRaw(const std::vector<netData>& inp, const netData& outp) {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		LastLayer->FeedForward(++Version);
		return TrainFnc(outp);
	}

	float NetContainer::Error(const netData& inp, const netData& outp) const {
		Inputs[0]->Input(inp);
		LastLayer->FeedForward(++Version);
		return GetErrorFnc(outp);
	}
	float NetContainer::ErrorRaw(const netData& inp, const netData& outp) const {
		Inputs[0]->InputRaw(inp);
		LastLayer->FeedForward(++Version);
		return GetErrorFnc(outp);
	}
	float NetContainer::Error(const std::vector<netData>& inp, const netData& outp) const {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->Input(inp[i]);
		LastLayer->FeedForward(++Version);
		return GetErrorFnc(outp);
	}
	float NetContainer::ErrorRaw(const std::vector<netData>& inp, const netData& outp) const {
		for (auto i = 0; i<inp.size(); i++)
			Inputs[i]->InputRaw(inp[i]);
		LastLayer->FeedForward(++Version);
		return GetErrorFnc(outp);
	}


	float NetContainer::Train_MeanSquare(const netData& TrainData) {
		netData dif = TrainData - output;
		Eigen::Map<Eigen::RowVectorXf> vec(dif.data(), dif.size());
		error = (0.5f * (vec * vec.transpose())(0)) / dif.cols();
		LastLayer->BackPropagate(dif);
		LastLayer->UpdateWeights();
		return error;
	}
	float NetContainer::GetError_MeanSquare(const netData& Data) {
		netData dif = Data - output;
		Eigen::Map<Eigen::RowVectorXf> vec(dif.data(), dif.size());
		return (0.5f * (vec * vec.transpose())(0)) / dif.cols();
	}

	float NetContainer::Train_SoftMax(const netData& TrainData) {
		float Error = -(TrainData.array()* Eigen::log(output.array())).sum() / output.cols();
		LastLayer->BackPropagate(TrainData - output);
		LastLayer->UpdateWeights();
		return Error;
	}
	float NetContainer::GetError_SoftMax(const netData& Data) {
		return -(Data.array()* Eigen::log(output.array())).sum() / output.cols();
	}

	void NetContainer::FeedForwardMeanSquare() {
		LastLayer->FeedForward(++Version);
		auto tmp = LastLayer->Output();
		output = outPart(tmp);
	}
	void NetContainer::FeedForwardSoftMax() {
		LastLayer->FeedForward(++Version);
		auto tmp2 = LastLayer->Output();
		netData tmp = Eigen::exp(outPart(tmp2).array());
		netData a = 1.0f / (netData::Ones(1, tmp.rows()) * tmp).array();
		netData div = Eigen::Map<Eigen::RowVectorXf>(a.data(), a.size()).asDiagonal();
		output = (tmp * div);
	}
	void NetContainer::FeedForwardStableSoftMax() {
		LastLayer->FeedForward(++Version);
		netData tmp = LastLayer->Output();
		output = outPart(tmp);

		for (auto i = 0; i < output.cols(); i++) {
			auto r = output.col(i);
			r -= netData::Constant(r.rows(), 1, r.maxCoeff());
			r = Eigen::exp(r.array());
			r *= 1.0f / r.sum();
		}
	}

	netData Reorder2(const netData& data, const size_t* Order, size_t count) {
		netData ret(data.rows(), count);
		for (auto i = 0; i<count; i++)
			ret.col(i) = data.col(Order[i]);
		return ret;
	}
	void NetContainer::SetBatchSize(int sz) {
		tmpOrder.resize(sz);
	}
	float NetContainer::Iterate() {
		if (tmpOrder.size()<TrainOrder.size()) {
			size_t i;
			for (i = 0; i<tmpOrder.size() && curIteration<TrainOrder.size(); i++, curIteration++) {
				tmpOrder[i] = TrainOrder[curIteration];
			}
			if (i<tmpOrder.size()) {
				std::random_device rd;
				std::mt19937 g(rd());
				std::shuffle(TrainOrder.begin(), TrainOrder.end(), g);
				curIteration = 0;

				for (; i<tmpOrder.size() && curIteration<TrainOrder.size(); i++, curIteration++) {
					tmpOrder[i] = TrainOrder[curIteration];
				}
			}

			for (auto i = 0; i<Inputs.size(); i++) {
				Inputs[i]->InputRaw(Reorder2(AllInputData[i], tmpOrder.data(), tmpOrder.size()));
			}
			FeedForward();
			return TrainFnc(Reorder2(AllTrainData, tmpOrder.data(), tmpOrder.size()));
		} else {
			for (auto i = 0; i<Inputs.size(); i++) {
				Inputs[i]->InputRaw(AllInputData[i]);
			}
			FeedForward();
			return TrainFnc(AllTrainData);
		}
	}


	void NetContainer::AddTrainDataRaw(const netData& inp, const netData& outp) {
		AddTrainDataRaw(std::vector<netData>{inp}, outp);
	}

	template<typename Container>
	void ForEach2(const Container& Data, std::function<void(decltype(Data.front())&, size_t)> fnc) {
		size_t i = 0;
		for (auto& it: Data)fnc(it, i++);
	}
	void NetContainer::AddTrainDataRaw(const std::vector<netData>& inp, const netData& outp) {
		curIteration = 0;
		if (!AllInputData.size()) {
			AllInputData.reserve(inp.size());
			for (auto& i : inp)
				AllInputData.push_back(i);
			AllTrainData = outp;
		} else {
			ForEach2(inp, [&](auto& item, auto i) {
				auto& t = AllInputData[i];
				auto newInput = netData(item.rows(), t.cols() + item.cols());
				newInput << t, item;
				t = newInput;
			});
			auto newOut = netData(outp.rows(), outp.cols() + AllTrainData.cols());
			newOut << AllTrainData, outp;
			AllTrainData = newOut;
		}

		auto next = TrainOrder.size();
		TrainOrder.resize(next + outp.cols());
		for (; next<TrainOrder.size(); next++)
			TrainOrder[next] = next;

		{
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(TrainOrder.begin(), TrainOrder.end(), g);
		}
	}

	void NetContainer::AddTrainData(const netData& inp, const netData& outp) {
		AddTrainData(std::vector<netData>{inp}, outp);
	}
	void NetContainer::AddTrainData(const std::vector<netData>& inp, const netData& outp) {
		curIteration = 0;
		if (!AllInputData.size()) {
			AllInputData.reserve(inp.size());
			size_t c = 0;
			for (auto& i : inp)
				AllInputData.push_back(Inputs[c++]->MakeRawInput(i));
			AllTrainData = outp;
		} else {
			ForEach2(inp, [&](auto& item, auto i) {
				auto& t = AllInputData[i];
				auto newInput = netData(item.rows() + 1, t.cols() + item.cols());
				newInput << t, Inputs[i]->MakeRawInput(item);
				t = newInput;
			});
			auto newOut = netData(outp.rows(), outp.cols() + AllTrainData.cols());
			newOut << AllTrainData, outp;
			AllTrainData = newOut;
		}

		auto next = TrainOrder.size();
		TrainOrder.resize(next + outp.cols());
		for (; next<TrainOrder.size(); next++)
			TrainOrder[next] = next;

		{
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(TrainOrder.begin(), TrainOrder.end(), g);
		}
	}

	netData NetContainer::FromVector(const std::vector<float>& Data) {
		auto r = LastLayer->nOutput();
		netData ret(r, Data.size()/r);
		memcpy(ret.data(), Data.data(), Data.size() * sizeof(float));
		return ret;
	}
};