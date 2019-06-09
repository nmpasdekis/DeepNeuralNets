#include <PVX_NeuralNetsCPU.h>
#include <random>

namespace PVX::DeepNeuralNets {
	netData Reorder(const netData& data, const int* Order, size_t count) {
		netData ret(data.rows(), count);
		for (auto i = 0; i<count; i++)
			ret.col(i) = data.col(Order[i]);
		return ret;
	}

	void NeuralNetContainer::AddTrainData(const netData& inp, const netData& outp) {
		AddTrainData(std::vector<netData>{inp}, outp);
	}

	template<typename Container>
	void ForEach(const Container& Data, std::function<void(decltype(Data.front())&, size_t)> fnc) {
		size_t i = 0;
		for (auto& it: Data)fnc(it, i++);
	}

	void NeuralNetContainer::AddTrainData(const std::vector<netData>& inp, const netData& outp) {
		std::random_device rd;
		std::mt19937 g(rd());

		ForEach(inp, [&](auto& item, auto i) {
			auto& t = InputData[i];
			auto newInput = netData(t.rows(), t.cols() + item.cols());
			newInput << t, item;
			t = newInput;
		});

		auto next = TrainOrder.size();
		TrainOrder.resize(next + outp.size());
		for (; next<TrainOrder.size(); next++)
			TrainOrder[next] = next;
		std::shuffle(TrainOrder.begin(), TrainOrder.end(), g);
	}
}