#include <PVX_NeuralNetsCPU.h>
#include <iostream>

namespace PVX {
	namespace DeepNeuralNets {
		static Eigen::MatrixXf makeComb(const std::vector<NeuralLayer_Base*> & inputs) {
			int count = 0;
			for (auto i : inputs) count += i->nOutput();
			return Eigen::MatrixXf::Zero(count + 1, 1);
		}

		void NeuronCombiner::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
			bin.Begin("CMBN");
			{
				for (auto& i: InputLayers)
					bin.write(int(IndexOf.at(i)));
			}
			bin.End();
		}

		void NeuronCombiner::DNA(std::map<void*, WeightData>& Weights) {
			for (auto l : InputLayers)
				l->DNA(Weights);
		}

		NeuronCombiner::NeuronCombiner(const int inputs) {
			output = Eigen::MatrixXf::Zero(inputs + 1, 1);
		}

		NeuronCombiner::NeuronCombiner(const std::vector<NeuralLayer_Base*> & inputs) {
			for (auto& i:inputs)InputLayers.push_back(i);
			output = makeComb(inputs);
		}
		void NeuronCombiner::FeedForward(int Version) {
			if (Version > FeedVersion) {
				InputLayers[0]->FeedForward(Version);
				int Start = 0;
				if (InputLayers[0]->BatchSize() != output.cols()) {
					output.conservativeResize(Eigen::NoChange, InputLayers[0]->BatchSize());
				}
				{
					const auto & o = InputLayers[0]->Output();
					output.block(Start, 0, o.rows(), o.cols()) = o;
					Start += o.rows() - 1;
				}
				for (auto i = 1; i < InputLayers.size();i++) {
					InputLayers[i]->FeedForward(Version);
					const auto & o = InputLayers[i]->Output();
					output.block(Start, 0, o.rows(), o.cols()) = o;
					Start += o.rows() - 1;
				}
				FeedVersion = Version;
			}
		}
		void NeuronCombiner::BackPropagate(const Eigen::MatrixXf & Gradient) {
			int Start = 0;
			for (auto i : InputLayers) {
				i->BackPropagate(Gradient.block(Start, 0, i->nOutput(), Gradient.cols()));
				Start += i->nOutput();
			}
		}

		size_t NeuronCombiner::nInput() {
			return output.rows() - 1;
		}

		void NeuronCombiner::Save(PVX::BinSaver & bin) {
			bin.Begin("COMB"); {
				for (auto inp : InputLayers)
					inp->Save(bin);
			}bin.End();
		}

		void NeuronCombiner::Load(PVX::BinLoader & bin) {
			int i = 0;
			bin.ProcessAny([&i, this](PVX::BinLoader & bin2, auto h) {
				InputLayers[i++]->Load(bin2);
			});
		}
	}
}