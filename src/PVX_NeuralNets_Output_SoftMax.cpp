#include <PVX_NeuralNetsCPU.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX {
	namespace DeepNeuralNets {
		SoftmaxOutput::SoftmaxOutput(NeuralLayer_Base * Last) : NeuralNetOutput{ Last } {}
		float SoftmaxOutput::Train(const float * Data) {
			return Train(Eigen::Map<Eigen::MatrixXf>((float*)Data, 1, output.cols(), Eigen::Stride<0, 0>()));
		}
		float SoftmaxOutput::Train(const Eigen::MatrixXf & TrainData) {
			float curErr = -(TrainData.array()* Eigen::log(output.array())).sum() / output.cols();
			if (Error >= 0)
				Error = Error * 0.99f + 0.01f * curErr;
			else
				Error = curErr;

			LastLayer->BackPropagate(TrainData - output);
			return Error;
		}

		void SoftmaxOutput::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
			bin.Write("SFTM", int(IndexOf.at(LastLayer)));
		}

		void SoftmaxOutput::FeedForward() {
			LastLayer->FeedForward(++Version);
			auto tmp2 = LastLayer->Output();
			Eigen::MatrixXf tmp = Eigen::exp(outPart(tmp2).array());
			Eigen::MatrixXf a = 1.0f / (Eigen::MatrixXf::Ones(1, tmp.rows()) * tmp).array();
			Eigen::MatrixXf div = Eigen::Map<Eigen::RowVectorXf>(a.data(), a.size()).asDiagonal();
			output = (tmp * div);
		}

		SoftmaxOutput::SoftmaxOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*>& Prevs) :NeuralNetOutput(Prevs.at(bin.read<int>()-1)) {}

		StableSoftmaxOutput::StableSoftmaxOutput(NeuralLayer_Base * Last) : NeuralNetOutput{ Last } {}
		float StableSoftmaxOutput::Train(const float * Data) {
			return Train(Eigen::Map<Eigen::MatrixXf>((float*)Data, 1, output.cols(), Eigen::Stride<0, 0>()));
		}
		float StableSoftmaxOutput::Train(const Eigen::MatrixXf & TrainData) {
			float curErr = -(TrainData.array()* Eigen::log(output.array())).sum() / output.cols();
			if (Error >= 0)
				Error = Error * 0.99f + 0.01f * curErr;
			else
				Error = curErr;

			LastLayer->BackPropagate(TrainData - output);
			return Error;
		}


		void StableSoftmaxOutput::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
			bin.Write("SSFM", int(IndexOf.at(LastLayer)));
		}

		void StableSoftmaxOutput::FeedForward() {
			LastLayer->FeedForward(++Version);
			Eigen::MatrixXf tmp = LastLayer->Output();
			output = outPart(tmp);

			for (auto i = 0; i < output.cols(); i++) {
				auto r = output.col(i);
				r -= Eigen::MatrixXf::Constant(r.rows(), 1, r.maxCoeff());
				r = Eigen::exp(r.array());
				r *= 1.0f / r.sum();
			}
		}

		StableSoftmaxOutput::StableSoftmaxOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*>& Prevs) :NeuralNetOutput(Prevs.at(bin.read<int>()-1)) {}

		float SoftmaxOutput::GetError(const Eigen::MatrixXf& Data) {
			Eigen::MatrixXf dif = Data - output;
			Eigen::Map<Eigen::RowVectorXf> vec(dif.data(), dif.size());
			return (0.5f * (vec * vec.transpose())(0)) / dif.cols();
		}

		float StableSoftmaxOutput::GetError(const Eigen::MatrixXf& Data) {
			Eigen::MatrixXf dif = Data - output;
			Eigen::Map<Eigen::RowVectorXf> vec(dif.data(), dif.size());
			return (0.5f * (vec * vec.transpose())(0)) / dif.cols();
		}

	}
}