#include <PVX_NeuralNetsCPU.h>
#include <random>

namespace PVX::DeepNeuralNets {
	netData Reorder(const netData& data, const int* Order, size_t count) {
		netData ret(data.rows(), count);
		for (auto i = 0; i<count; i++)
			ret.col(i) = data.col(Order[i]);
		return ret;
	}

	void NeuralNetContainer::AddTrainDataRaw(const netData& inp, const netData& outp) {
		AddTrainDataRaw(std::vector<netData>{inp}, outp);
	}

	template<typename Container>
	void ForEach(const Container& Data, std::function<void(decltype(Data.front())&, size_t)> fnc) {
		size_t i = 0;
		for (auto& it: Data)fnc(it, i++);
	}

	void NeuralNetContainer::AddTrainDataRaw(const std::vector<netData>& inp, const netData& outp) {
		curIteration = 0;
		if (!AllInputData.size()) {
			AllInputData.reserve(inp.size());
			for (auto& i : inp)
				AllInputData.push_back(i);
			AllTrainData = outp;
		} else {
			ForEach(inp, [&](auto& item, auto i) {
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
	void NeuralNetContainer::SetBatchSize(int sz) {
		tmpOrder.resize(sz);
	}
	float NeuralNetContainer::Iterate() {
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
				Inputs[i]->InputRaw(Reorder(AllInputData[i], tmpOrder.data(), tmpOrder.size()));
			}
			Output->FeedForward();
			return Output->Train(Reorder(AllTrainData, tmpOrder.data(), tmpOrder.size()));
		} else {
			for (auto i = 0; i<Inputs.size(); i++) {
				Inputs[i]->InputRaw(AllInputData[i]);
			}
			Output->FeedForward();
			return Output->Train(AllTrainData);
		}
	}
}