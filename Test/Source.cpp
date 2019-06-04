#include<PVX_NeuralNetsCPU.h>
#include<PVX_GenericSolvers.h>
#include<iostream>

using namespace PVX::DeepNeuralNets;
using namespace PVX::Solvers;

using netData = Eigen::MatrixXf;

float Poly(float x, const std::vector<float>& Factors) {
	float y = Factors[0];
	for (size_t i = 1; i<Factors.size(); i++) y = y * x + Factors[i];
	return y;
}



int main() {

	return 0;
}

int main2() {
	NeuralNetContainer net(L"myXor.nn");

	netData InputData = net.MakeRawInput({
		1.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f
	});
	std::cout << InputData << "\n\n";

	//auto res = net.ProcessRaw(InputData);
	//std::cout << res << "\n\n";

	auto res = net.ProcessRaw(InputData.col(0));
	std::cout << res << "\n\n";

	res = net.ProcessRaw(InputData.col(1));
	std::cout << res << "\n\n";

	res = net.ProcessRaw(InputData.col(2));
	std::cout << res << "\n\n";

	res = net.ProcessRaw(InputData.col(3));
	std::cout << res << "\n\n";


	return 0;
}

int main3() {
	InputLayer Input("Input", 2);
	NeuronLayer Hidden1("Hidden1", &Input, 10);
	NeuronLayer Hidden2("Hidden2", &Hidden1, 10);
	NeuronLayer Last("Last", &Hidden2, 2);
	MeanSquareOutput Output(&Last);
	NeuralNetContainer Network(&Output);

	netData InputData = Input.MakeRawInput({
		1.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f
	});

	netData TrainData = Network.FromVector({
		1.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,
		0.0f, 1.0f
	});

	float Error = 1.0f;
	int iter = 0;
	while (Error>1e-13) {
		Error = Network.TrainRaw(InputData, TrainData);
		if (!(iter++%1000)) {
			auto r = Network.ProcessRaw(InputData);
			std::cout << "\r" << Error << "                              ";
		}
	}

	std::cout << "\n" << Error << "\n\n" << Network.ProcessRaw(InputData) << "\n\nI know Kung Fu!!!\n\n";

	Network.Save(L"myXor.nn");

	return 0;
}