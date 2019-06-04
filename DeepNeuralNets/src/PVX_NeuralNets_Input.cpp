#include <PVX_NeuralNetsCPU.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX {
	namespace DeepNeuralNets {
		void InputLayer::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
			bin.Begin("INPT"); {
				if (name.size()) bin.Write("NAME", name);
				bin.Write("ICNT", int(nInput()));
			} bin.End();
		}
		InputLayer::InputLayer(PVX::BinLoader& bin){
			int ic;
			bin.Read("NAME", name);
			bin.Read("ICNT", ic);
			bin.Execute();
			output = Eigen::MatrixXf::Ones(ic + 1ll, 1ll);
		}
		InputLayer::InputLayer(const size_t Size) {
			output = Eigen::MatrixXf::Ones(Size + 1, 1);
		}
		InputLayer::InputLayer(const std::string& Name, const size_t Size) {
			name = Name;
			output = Eigen::MatrixXf::Ones(Size + 1ll, 1ll);
		}


		int InputLayer::Input(const float * Data, int Count) {
			if (output.cols() != Count)
				output = Eigen::MatrixXf::Ones(output.rows(), Count);
			outPart(output) = Map((float*)Data, output.rows() - 1ll, output.cols());
			return 1;
		}

		int InputLayer::Input(const Eigen::MatrixXf & Data) {
			if (output.rows() == Data.rows() + 1) {
				if (output.cols() != Data.cols()) {
					output = Eigen::MatrixXf::Ones(output.rows(), Data.cols());
				}
				outPart(output) = Data;
				return 1;
			}
			return 0;
		}

		void InputLayer::InputRaw(const Eigen::MatrixXf & Data) {
			output = Data;
		}

		Eigen::MatrixXf InputLayer::MakeRawInput(const Eigen::MatrixXf & Data) {
			Eigen::MatrixXf ret = Eigen::MatrixXf::Ones(Data.rows() + 1, Data.cols());
			outPart(ret) = Data;
			return ret;
		}

		Eigen::MatrixXf InputLayer::MakeRawInput(const std::vector<float>& Input) {
			return MakeRawInput(Input.data(), Input.size() / nInput());
		}

		Eigen::MatrixXf InputLayer::MakeRawInput(const float * Data, int Count) {
			return MakeRawInput(Eigen::Map<Eigen::MatrixXf>((float*)Data, output.rows() - 1, Count));
		}

		size_t InputLayer::nInput() {
			return output.rows() - 1;
		}
	}
}