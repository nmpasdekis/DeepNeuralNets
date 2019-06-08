#include <PVX_NeuralNetsCPU_Depricated.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX {
	namespace DeepNeuralNets {
		MeanSquareOutput::MeanSquareOutput(NeuralLayer_Base * Last) : NeuralNetOutput_Base{ Last } {}
		float MeanSquareOutput::Train(const float * Data) {
			return Train(Eigen::Map<Eigen::MatrixXf>((float*)Data, 1, output.cols()));;
		}
		float MeanSquareOutput::GetError(const Eigen::MatrixXf & Data) {
			Eigen::MatrixXf dif = Data - output;
			Eigen::Map<Eigen::RowVectorXf> vec(dif.data(), dif.size());
			return (0.5f * (vec * vec.transpose())(0)) / dif.cols();
		}
		float MeanSquareOutput::Train(const Eigen::MatrixXf & TrainData) {
			Eigen::MatrixXf dif = TrainData - output;

			Eigen::Map<Eigen::RowVectorXf> vec(dif.data(), dif.size());

			float curErr = (0.5f * (vec * vec.transpose())(0)) / dif.cols();
			if (Error >= 0)
				Error = Error * 0.9f + 0.1f * curErr;
			else
				Error = curErr;
			LastLayer->BackPropagate(dif);
			return Error;
		}
		float MeanSquareOutput::Train2(const Eigen::MatrixXf & TrainData) {
			Eigen::MatrixXf dif = TrainData - output;

			Eigen::Map<Eigen::RowVectorXf> vec(dif.data(), dif.size());

			Error = (0.5f * (vec * vec.transpose())(0)) / dif.cols();
			return Error;
		}

		void MeanSquareOutput::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
			bin.Write("MSQR", int(IndexOf.at(LastLayer)));
		}
		MeanSquareOutput::MeanSquareOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*>& Prevs) :NeuralNetOutput_Base(Prevs.at(bin.read<int>()-1)) {}
	}
}