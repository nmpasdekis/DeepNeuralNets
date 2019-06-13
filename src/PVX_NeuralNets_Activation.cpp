#include <PVX_NeuralNetsCPU.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX {
	namespace DeepNeuralNets {
				/////////////////////////////////////

		static netData Tanh(const netData& x) {
			return Eigen::tanh(x.array());
		}
		static netData TanhDer(const netData& x) {
			auto tmp = Eigen::tanh(x.array());
			return 1.0f - tmp * tmp;
		}
		static netData TanhBias(const netData& x) {
			netData tmp = Eigen::tanh(x.array());
			auto dt = tmp.data();
			size_t sz = x.cols()*x.rows();
			for (auto i = 0; i < sz; i++)
				dt[i] = dt[i] * 0.5f + 0.5f;
			return tmp;
		}
		static netData TanhBiasDer(const netData & x) {
			auto tmp = Eigen::tanh(x.array());
			return 0.5f* (1.0f - tmp * tmp);
		}
		static netData Relu(const netData & x) {
			return x.array()* (x.array() > 0).cast<float>();
		}
		static netData ReluDer(const netData & x) {
			netData ret(x.rows(), x.cols());
			float* dt = (float*)x.data();
			float* o = ret.data();
			size_t sz = x.cols()*x.rows();
			for (int i = 0; i < sz; i++) o[i] = (dt[i] > 0) ? 1.0f : 0.0001f;
			return ret;
		}
		static netData Sigmoid(const netData & x) {
			netData ex = Eigen::exp(-x.array());
			netData ret = 1.0f / (1.0f + ex.array());
			return ret;
		}
		static netData SigmoidDer(const netData & x) {
			netData tmp = Sigmoid(x);
			return tmp.array()* (1.0f - tmp.array());
		}
		static netData Linear(const netData & x) {
			return x;
		}
		static netData LinearDer(const netData & x) {
			return netData::Ones(x.rows(), x.cols());
		}

		////////////////////////////////////

		ActivationLayer::ActivationLayer(NeuralLayer_Base* inp, LayerActivation Activation) :
			ActivationLayer(inp->nOutput(), Activation) {
			PreviousLayer = inp;
		}
		ActivationLayer::ActivationLayer(int inp, LayerActivation Activation) : activation{ Activation } {
			PreviousLayer = nullptr;
			output = netData::Ones(inp + 1ll, 1);
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
					output = netData::Ones(output.rows(), inp.cols());
				}
				
				outPart(output) = Activate(inp);
				FeedVersion = Version;
			}
		}
		void ActivationLayer::BackPropagate(const netData& Gradient) {
			netData grad = Gradient.array() * Derivative(outPart(output)).array();
			PreviousLayer->BackPropagate(outPart(grad));
		}

		void ActivationLayer::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const {
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
		size_t ActivationLayer::nInput() const {
			return nOutput();
		}
	}
}