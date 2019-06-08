#include <PVX_NeuralNetsCPU.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX {
	namespace DeepNeuralNets {
		OutputLayer::OutputLayer(NeuralLayer_Base * Last, OutputType Type) :LastLayer{ Last }, Type{ Type } {
			switch (Type) {
				case PVX::DeepNeuralNets::OutputType::MeanSquare:
					FeedForward = [this] { FeedForwardMeanSquare(); };
					GetErrorFnc = [this](const Eigen::MatrixXf& Data) { return GetError_MeanSquare(Data); };
					TrainFnc = [this](const Eigen::MatrixXf& Data) { return Train_MeanSquare(Data); };
					break;
				case PVX::DeepNeuralNets::OutputType::SoftMax:
					FeedForward = [this] { FeedForwardSoftMax(); };
					GetErrorFnc = [this](const Eigen::MatrixXf& Data) { return GetError_SoftMax(Data); };
					TrainFnc = [this](const Eigen::MatrixXf& Data) { return Train_SoftMax(Data); };
					break;
				case PVX::DeepNeuralNets::OutputType::StableSoftMax:
					FeedForward = [this] { FeedForwardStableSoftMax(); };
					GetErrorFnc = [this](const Eigen::MatrixXf& Data) { return GetError_SoftMax(Data); };
					TrainFnc = [this](const Eigen::MatrixXf& Data) { return Train_SoftMax(Data); };
					break;
				default:
					break;
			}
		}
		float OutputLayer::GetError(const Eigen::MatrixXf & Data) {
			return GetErrorFnc(Data);
		}
		float OutputLayer::Train(const Eigen::MatrixXf & Data) {
			return TrainFnc(Data);
		}
		float OutputLayer::Train(const float * Data) {
			return Train(Eigen::Map<Eigen::MatrixXf>((float*)Data, 1, output.cols()));;
		}

		float OutputLayer::Train_MeanSquare(const Eigen::MatrixXf & TrainData) {
			Eigen::MatrixXf dif = TrainData - output;
			Eigen::Map<Eigen::RowVectorXf> vec(dif.data(), dif.size());
			Error = (0.5f * (vec * vec.transpose())(0)) / dif.cols();
			LastLayer->BackPropagate(dif);
			return Error;
		}
		float OutputLayer::GetError_MeanSquare(const Eigen::MatrixXf & Data) {
			Eigen::MatrixXf dif = Data - output;
			Eigen::Map<Eigen::RowVectorXf> vec(dif.data(), dif.size());
			return (0.5f * (vec * vec.transpose())(0)) / dif.cols();
		}

		float OutputLayer::Train_SoftMax(const Eigen::MatrixXf & TrainData) {
			float Error = -(TrainData.array()* Eigen::log(output.array())).sum() / output.cols();
			LastLayer->BackPropagate(TrainData - output);
			return Error;
		}
		float OutputLayer::GetError_SoftMax(const Eigen::MatrixXf& Data) {
			return -(Data.array()* Eigen::log(output.array())).sum() / output.cols();
		}

		void OutputLayer::FeedForwardMeanSquare() {
			LastLayer->FeedForward(++Version);
			auto tmp = LastLayer->Output();
			output = outPart(tmp);
		}
		void OutputLayer::FeedForwardSoftMax() {
			LastLayer->FeedForward(++Version);
			auto tmp2 = LastLayer->Output();
			Eigen::MatrixXf tmp = Eigen::exp(outPart(tmp2).array());
			Eigen::MatrixXf a = 1.0f / (Eigen::MatrixXf::Ones(1, tmp.rows()) * tmp).array();
			Eigen::MatrixXf div = Eigen::Map<Eigen::RowVectorXf>(a.data(), a.size()).asDiagonal();
			output = (tmp * div);
		}
		void OutputLayer::FeedForwardStableSoftMax() {
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
		void OutputLayer::Result(float* res) {
			auto r = Result();
			memcpy(res, r.data(), sizeof(float) * r.cols());
		}

		const Eigen::MatrixXf& OutputLayer::Result() {
			FeedForward();
			return output;
		}
		std::set<NeuralLayer_Base*> OutputLayer::Gather() {
			std::set<NeuralLayer_Base*> g;
			LastLayer->Gather(g);
			return g;
		}
		int OutputLayer::nOutput() {
			return LastLayer->output.rows() - 1;
		}
		void OutputLayer::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
			bin.Write("TYPE", int(Type));
			bin.Write("LAST", int(IndexOf.at(LastLayer)));
		}
	
		void OutputLayer::SaveCheckpoint() {
			if (Checkpoint.Layers.size()==0) {
				Checkpoint = GetDNA();
			}
			CheckpointDNA = Checkpoint.GetData();
			CheckpointError = Error;
		}

		float OutputLayer::LoadCheckpoint() {
			Error = CheckpointError;
			Checkpoint.SetData(CheckpointDNA.data());
			return Error;
		}

		void OutputLayer::ResetMomentum() {
			LastLayer->ResetMomentum();
		}

		NetDNA OutputLayer::GetDNA() {
			std::map<void*, WeightData> data;
			LastLayer->DNA(data);
			NetDNA ret;
			ret.Size = 0;
			for (auto&[l, dt] : data) {
				dt.Offset = ret.Size;
				ret.Size += dt.Count;
				ret.Layers.push_back(dt);
			}
			return ret;
		}
	}
}