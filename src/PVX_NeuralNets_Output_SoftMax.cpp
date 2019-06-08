#include <PVX_NeuralNetsCPU_Depricated.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX {
	namespace DeepNeuralNets {
		SoftmaxOutput::SoftmaxOutput(NeuralLayer_Base * Last) : NeuralNetOutput_Base{ Last } {}
		float SoftmaxOutput::Train(const float * Data) {
			return Train(Eigen::Map<netData>((float*)Data, 1, output.cols(), Eigen::Stride<0, 0>()));
		}
		float SoftmaxOutput::Train(const netData & TrainData) {
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
			netData tmp = Eigen::exp(outPart(tmp2).array());
			netData a = 1.0f / (netData::Ones(1, tmp.rows()) * tmp).array();
			netData div = Eigen::Map<Eigen::RowVectorXf>(a.data(), a.size()).asDiagonal();
			output = (tmp * div);
		}

		SoftmaxOutput::SoftmaxOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*>& Prevs) :NeuralNetOutput_Base(Prevs.at(bin.read<int>()-1)) {}

		StableSoftmaxOutput::StableSoftmaxOutput(NeuralLayer_Base * Last) : NeuralNetOutput_Base{ Last } {}
		float StableSoftmaxOutput::Train(const float * Data) {
			return Train(Eigen::Map<netData>((float*)Data, 1, output.cols(), Eigen::Stride<0, 0>()));
		}
		float StableSoftmaxOutput::Train(const netData & TrainData) {
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
			netData tmp = LastLayer->Output();
			output = outPart(tmp);

			for (auto i = 0; i < output.cols(); i++) {
				auto r = output.col(i);
				r -= netData::Constant(r.rows(), 1, r.maxCoeff());
				r = Eigen::exp(r.array());
				r *= 1.0f / r.sum();
			}
		}

		StableSoftmaxOutput::StableSoftmaxOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*>& Prevs) :NeuralNetOutput_Base(Prevs.at(bin.read<int>()-1)) {}

		float SoftmaxOutput::GetError(const netData& Data) {
			return -(Data.array()* Eigen::log(output.array())).sum() / output.cols();
		}

		float StableSoftmaxOutput::GetError(const netData& Data) {
			return -(Data.array()* Eigen::log(output.array())).sum() / output.cols();
		}

	}
}