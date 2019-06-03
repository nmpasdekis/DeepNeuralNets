#include <PVX_NeuralNetsCPU.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX {
	namespace DeepNeuralNets {
				/////////////////////////////////////

		static Eigen::MatrixXf Tanh(const Eigen::MatrixXf& x) {
			return Eigen::tanh(x.array());
		}
		static Eigen::MatrixXf TanhDer(const Eigen::MatrixXf& x) {
			auto tmp = Eigen::tanh(x.array());
			return 1.0f - tmp * tmp;
		}
		static Eigen::MatrixXf TanhBias(const Eigen::MatrixXf& x) {
			Eigen::MatrixXf tmp = Eigen::tanh(x.array());
			auto dt = tmp.data();
			size_t sz = x.cols()*x.rows();
			for (auto i = 0; i < sz; i++)
				dt[i] = dt[i] * 0.5f + 0.5f;
			return tmp;
		}
		static Eigen::MatrixXf TanhBiasDer(const Eigen::MatrixXf & x) {
			auto tmp = Eigen::tanh(x.array());
			return 0.5f* (1.0f - tmp * tmp);
		}
		static Eigen::MatrixXf Relu(const Eigen::MatrixXf & x) {
			return x.array()* (x.array() > 0).cast<float>();
		}
		static Eigen::MatrixXf ReluDer(const Eigen::MatrixXf & x) {
			Eigen::MatrixXf ret(x.rows(), x.cols());
			float* dt = (float*)x.data();
			float* o = ret.data();
			size_t sz = x.cols()*x.rows();
			for (int i = 0; i < sz; i++) o[i] = (dt[i] > 0) ? 1.0f : 0.0001f;
			return ret;
		}
		static Eigen::MatrixXf Sigmoid(const Eigen::MatrixXf & x) {
			Eigen::MatrixXf ex = Eigen::exp(-x.array());
			Eigen::MatrixXf ret = 1.0f / (1.0f + ex.array());
			return ret;
		}
		static Eigen::MatrixXf SigmoidDer(const Eigen::MatrixXf & x) {
			Eigen::MatrixXf tmp = Sigmoid(x);
			return tmp.array()* (1.0f - tmp.array());
		}
		static Eigen::MatrixXf Linear(const Eigen::MatrixXf & x) {
			return x;
		}
		static Eigen::MatrixXf LinearDer(const Eigen::MatrixXf & x) {
			return Eigen::MatrixXf::Ones(x.rows(), x.cols());
		}

		////////////////////////////////////

		ActivationLayer::ActivationLayer(NeuralLayer_Base* inp, LayerActivation Activation) :
			ActivationLayer(inp->nOutput(), Activation) {
			PreviousLayer = inp;
		}
		ActivationLayer::ActivationLayer(int inp, LayerActivation Activation) : activation{ Activation } {
			PreviousLayer = nullptr;
			output = Eigen::MatrixXf{ inp + 1, 1 };
			switch (Activation) {
				case LayerActivation::Tanh:
					Activate = Tanh;
					Derivative = TanhDer;
					break;
				case LayerActivation::TanhBias:
					Activate = TanhBias;
					Derivative = TanhBiasDer;
					break;
				case LayerActivation::ReLU:
					Activate = Relu;
					Derivative = ReluDer;
					break;
				case LayerActivation::Sigmoid:
					Activate = Sigmoid;
					Derivative = SigmoidDer;
					break;
				case LayerActivation::Linear:
					Activate = Linear;
					Derivative = LinearDer;
					break;
			}
		}
		void ActivationLayer::FeedForward(int Version) {
			if (Version > FeedVersion) {
				PreviousLayer->FeedForward(Version);
				auto inp = PreviousLayer->Output();
				if (inp.cols() != output.cols()) {
					output = Eigen::MatrixXf::Ones(output.rows(), inp.cols());
				}
				
				outPart(output) = Activate(inp);
				FeedVersion = Version;
			}
		}
		void ActivationLayer::BackPropagate(const Eigen::MatrixXf& Gradient) {
			Eigen::MatrixXf grad = Gradient.array() * Derivative(outPart(output)).array();
			PreviousLayer->BackPropagate(outPart(grad));
		}

		void ActivationLayer::Save(PVX::BinSaver& bin) {
			bin.Begin("ACTV"); {
				bin.Write("ACTV", (char)activation);
				bin.Begin("INPT"); {
					PreviousLayer->Save(bin);
				} bin.End();
			}bin.End();
		}
		void ActivationLayer::Load(PVX::BinLoader& bin) {
			bin.Process("ACTV", [this](PVX::BinLoader & bin2) {
				bin2.Process("INPT", [this](PVX::BinLoader & bin2) {
					PreviousLayer->Load(bin2);
				});
			});
		}

		void ActivationLayer::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
			bin.Begin("ACTV");
			{
				bin.Write("OUTC", nOutput());
				bin.Write("ACTV", int(activation));
				bin.Write("INPT", IndexOf.at(PreviousLayer));
			}
			bin.End();
		}

		ActivationLayer* ActivationLayer::Load2(PVX::BinLoader& bin) {
			int outc, act, prev;
			bin.Read("OUTC", outc);
			bin.Read("ACTV", act);
			bin.Read("INPT", prev);
			bin.Execute();
			auto ret = new ActivationLayer(outc, LayerActivation(act));
			*(int*)&ret->PreviousLayer = prev;
			return ret;
		}

		void ActivationLayer::DNA(std::map<void*, WeightData>& Weights) {
			PreviousLayer->DNA(Weights);
		}
		void ActivationLayer::SetLearnRate(float a) {
			PreviousLayer->SetLearnRate(a);
		}
		void ActivationLayer::ResetMomentum() {
			PreviousLayer->ResetMomentum();
		}
		size_t ActivationLayer::nInput() {
			return nOutput();
		}
	}
}