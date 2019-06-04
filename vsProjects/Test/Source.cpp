#include<PVX_NeuralNetsCPU.h>
#include<PVX_GenericSolvers.h>
#include<iostream>
#include <random>

using namespace PVX::DeepNeuralNets;
using namespace PVX::Solvers;

using netData = Eigen::MatrixXf;

float Poly(float x, const std::vector<float>& Factors) {
	float y = Factors[0];
	for (size_t i = 1; i<Factors.size(); i++) y = y * x + Factors[i];
	return y;
}

std::vector<float> Range(float from, float to, float step) {
	size_t sz = (to - from)/step;
	std::vector<float> ret(sz);
	for (auto& t : ret) {
		t = from;
		from += step;
	}
	return std::move(ret);
}

void Evaluate(std::vector<float>& out, const std::vector<float>& range, const std::vector<float>& Model) {
	out.resize(range.size());
#pragma omp parallel for
	for (auto i = 0; i<range.size(); i++) {
		out[i] = Poly(range[i], Model);
	}
}

void Print(const std::vector<float>& v) {
	for (auto f : v)
		printf("%5.2f ", f);
		//std::cout << f << " ";
}

int main() {
	auto TargetPoly = std::vector<float>{ 1.0f, -2.0f, 3.0f, -4.0f, 5.0f/*, -6.0f, 7.0f, -8.0f*/ };
	auto Model = std::vector<float>(TargetPoly.size());
	auto RealValues = std::vector<float>();
	auto TestValues = std::vector<float>();
	auto x = Range(-5.0f, 5.0f, 0.5f);
	Evaluate(RealValues, x, TargetPoly);

	std::default_random_engine eng;
	std::normal_distribution<float> dist;

	//for (auto& r : RealValues) {
	//	r += dist(eng)*0.01f;
	//}
	
	auto ErrFnc = [&] {
		Evaluate(TestValues, x, Model);

		float sum = 0;
#pragma omp parallel for reduction(+:sum)
		for (auto i = 0; i<TestValues.size(); i++) {
			float tmp = TestValues[i] - RealValues[i];
			sum += tmp*tmp;
		}

		return sum / TestValues.size();
	};

	auto grad = GradientDescent(ErrFnc, Model.data(), Model.size(), 0.0001f);
	auto gen = GeneticSolver(ErrFnc, Model.data(), Model.size(), 100, 20, 0.5f);

	gen.OnNewGeneration([&] {
		grad.ClearMomentum();
		grad.RecalculateError();
		float err = 0;
		for (int i = 0; i<1000; i++)
			err = grad.Iterate();
		return err;
	});

	float Error = 1.0f, LastError = 1.0f;

	while (Error>1e-5) {
		Error = gen.Iterate();
		if (Error < LastError) {
			Print(Model);
			std::cout << Error << " " << gen.BestId() << "\n";
		}
		LastError = Error;
	}

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