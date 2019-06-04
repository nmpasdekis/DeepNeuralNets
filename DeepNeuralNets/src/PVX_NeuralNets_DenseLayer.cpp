#include <PVX_NeuralNetsCPU.h>
#include "PVX_NeuralNets_Util.inl"

namespace PVX {
	namespace DeepNeuralNets {
		auto RandomBias(size_t r, size_t c) {
			return Eigen::MatrixXf::Random(r, c).array() * 0.5f + 0.5f;
		}

		int UseDropout = 0;
		/////////////////////////////////////

		static Eigen::MatrixXf Tanh(const Eigen::MatrixXf & x) {
			return Eigen::tanh(x.array());
		}
		static Eigen::MatrixXf TanhDer(const Eigen::MatrixXf & x) {
			auto tmp = Eigen::tanh(x.array());
			return 1.0f - tmp * tmp;
		}
		static Eigen::MatrixXf TanhBias(const Eigen::MatrixXf & x) {
			Eigen::MatrixXf tmp = Eigen::tanh(x.array());
			auto dt = tmp.data();
			size_t sz = x.cols()*x.rows();
			for (auto i = 0; i < sz; i++)
				dt[i] = dt[i] * 0.5f + 0.5f;
			return tmp;
		}
		static Eigen::MatrixXf TanhBiasDer(const Eigen::MatrixXf & x) {
			auto tmp = Eigen::tanh(x.array());
			return 0.5f * (1.0f - tmp * tmp);
		}
		static Eigen::MatrixXf Relu(const Eigen::MatrixXf & x) {
			return x.array() * (x.array() > 0).cast<float>();
		}
		static Eigen::MatrixXf ReluDer(const Eigen::MatrixXf & x) {
			Eigen::MatrixXf ret(x.rows(), x.cols());
			float * dt = (float*)x.data();
			float * o = ret.data();
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
			return tmp.array() * (1.0f - tmp.array());
		}
		static Eigen::MatrixXf Linear(const Eigen::MatrixXf & x) {
			return x;
		}
		static Eigen::MatrixXf LinearDer(const Eigen::MatrixXf & x) {
			return Eigen::MatrixXf::Ones(x.rows(), x.cols());
		}

		////////////////////////////////////

		void NeuronLayer::AdamF(const Eigen::MatrixXf & Gradient) {
			auto g1 = Gradient.array();
			if (RMSprop.cols() != Gradient.cols())RMSprop = Eigen::MatrixXf::Ones(Gradient.rows(), Gradient.cols());
			RMSprop = _RMSprop * RMSprop.array() + _iRMSprop * (g1*g1);
			Eigen::MatrixXf gr = g1 / ((Eigen::sqrt(RMSprop.array()) + 1e-8));

			DeltaWeights = (gr * PreviousLayer->Output().transpose() * (_LearnRate * _iMomentum)) + DeltaWeights * _Momentum;
			Weights += DeltaWeights;
		}
		void NeuronLayer::MomentumF(const Eigen::MatrixXf & Gradient) {
			DeltaWeights = (Gradient * PreviousLayer->Output().transpose() * (_LearnRate * _iMomentum)) + DeltaWeights * _Momentum;
			Weights += DeltaWeights;
		}
		void NeuronLayer::RMSpropF(const Eigen::MatrixXf & Gradient) {
			auto g1 = Gradient.array();
			if (RMSprop.cols() != Gradient.cols())
				RMSprop = Eigen::MatrixXf::Ones(Gradient.rows(), Gradient.cols());
			RMSprop = _RMSprop * RMSprop.array() + _iRMSprop * (g1*g1);
			Eigen::MatrixXf gr = g1 / (Eigen::sqrt(RMSprop.array()) + 1e-8);

			Weights += (gr * PreviousLayer->Output().transpose() * _LearnRate);
		}
		void NeuronLayer::SgdF(const Eigen::MatrixXf & Gradient) {
			Weights += (Gradient * PreviousLayer->Output().transpose() * _LearnRate);
		}
		void NeuronLayer::AdaGradF(const Eigen::MatrixXf & Gradient) {
			auto g1 = Gradient.array();
			if (RMSprop.cols() != Gradient.cols())
				RMSprop = Eigen::MatrixXf::Ones(Gradient.rows(), Gradient.cols());
			RMSprop = RMSprop.array() + (g1*g1);
			Eigen::MatrixXf gr = g1 / (Eigen::sqrt(RMSprop.array()) + 1e-8);

			Weights += (gr * PreviousLayer->Output().transpose() * _LearnRate);
		}
		void NeuronLayer::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) {
			bin.Begin("DENS");
			{
				if (name.size()) bin.Write("NAME", name);
				bin.Write("ROWS", int(Weights.rows()));
				bin.Write("COLS", int(Weights.cols()));
				bin.Write("WGHT", Weights.data(), Weights.size());
				bin.Write("RATE", _LearnRate);
				bin.Write("MMNT", _Momentum);
				bin.Write("RMSP", _RMSprop);
				bin.Write("DRPT", _Dropout);
				bin.Write("ACTV", (int)(activation));
				bin.Write("TRNS", (int)(training));
				bin.Write("INPT", (int)(IndexOf.at(PreviousLayer)));
			}
			bin.End();
		}
		NeuralLayer_Base* NeuronLayer::Load2(PVX::BinLoader& bin) {
			int rows = 0, cols = 0, act = 0, train = 0, prev = 0;
			float rate, rms, drop, momentum;
			std::string Name;
			std::vector<float> Weights;
			bin.Read("ROWS", rows);
			bin.Read("COLS", cols);
			bin.Read("WGHT", Weights);
			bin.Read("RATE", rate);
			bin.Read("MMNT", momentum);
			bin.Read("RMSP", rms);
			bin.Read("DRPT", drop);
			bin.Read("ACTV", act);
			bin.Read("TRNS", train);
			bin.Read("INPT", prev);
			bin.Read("NAME", Name);
			bin.Execute();
			auto ret = new NeuronLayer(cols-1, rows, LayerActivation(act), TrainScheme(train));
			if (Name.size()) ret->name = Name;
			ret->GetWeights() = Eigen::Map<Eigen::MatrixXf>(Weights.data(), rows, cols);
			ret->_Dropout = drop;
			ret->_iDropout = 1.0f / drop;
			ret->_Momentum = momentum;
			ret->_iMomentum = 1.0f - momentum;
			ret->_LearnRate = rate;
			ret->_RMSprop = rms;
			ret->_iRMSprop = 1.0f - rms;
			ret->PreviousLayer = reinterpret_cast<NeuralLayer_Base*>(prev);// (NeuralLayer_Base*)prev;
			return ret;
		}
		void NeuronLayer::SetLearnRate(float a) {
			_LearnRate = a;
			PreviousLayer->SetLearnRate(a);
		}
		Eigen::MatrixXf& NeuronLayer::GetWeights() {
			return Weights;
		}
		void NeuronLayer::ResetMomentum() {
			this->RMSprop = Eigen::MatrixXf::Ones(this->RMSprop.rows(), this->RMSprop.cols());
			this->DeltaWeights = Eigen::MatrixXf::Zero(this->DeltaWeights.rows(), this->DeltaWeights.cols());
			PreviousLayer->ResetMomentum();
		}
		static int InitOpenMP = 0;

		NeuronLayer::NeuronLayer(int nInput, int nOutput, LayerActivation Activation, TrainScheme Train, float WeightMax) :
			training{ Train },
			activation{ Activation },
			DeltaWeights{ Eigen::MatrixXf::Zero(nOutput, nInput + 1ll) },
			Weights{ Eigen::MatrixXf::Random(nOutput, nInput + 1ll) },
			RMSprop{ Eigen::MatrixXf::Ones(nOutput, 1ll) }
		{
			if (!InitOpenMP) {
				Eigen::initParallel();
				InitOpenMP = 1;
			}

			_LearnRate = __LearnRate;
			_Momentum = __Momentum;
			_iMomentum = __iMomentum;
			_RMSprop = __RMSprop;
			_iRMSprop = __iRMSprop;
			_Dropout = __Dropout;
			_iDropout = __iDropout;

			float randScale = sqrtf(2.0f / (nInput + 1));

			output = Eigen::MatrixXf::Ones(nOutput + 1, 1);
			switch (Activation) {
			case LayerActivation::Tanh:
				randScale = sqrtf(1.0f / (nInput + 1));
				Activate = Tanh;
				Derivative = TanhDer;
				break;
			case LayerActivation::TanhBias:
				randScale = sqrtf(1.0f / (nInput + 1));
				Activate = TanhBias;
				Derivative = TanhBiasDer;
				break;
			case LayerActivation::ReLU:
				Activate = Relu;
				Derivative = ReluDer;
				break;
			case LayerActivation::Sigmoid:
				randScale = sqrtf(1.0f / (nInput + 1));
				Activate = Sigmoid;
				Derivative = SigmoidDer;
				break;
			case LayerActivation::Linear:
				Activate = Linear;
				Derivative = LinearDer;
				break;
			}
			Weights *= randScale;
			switch (Train) {
			case TrainScheme::Adam: updateWeights = &NeuronLayer::AdamF; break;
			case TrainScheme::RMSprop: updateWeights = &NeuronLayer::RMSpropF; break;
			case TrainScheme::Momentum: updateWeights = &NeuronLayer::MomentumF; break;
			case TrainScheme::AdaGrad: updateWeights = &NeuronLayer::AdaGradF; break;
			case TrainScheme::Sgd: updateWeights = &NeuronLayer::SgdF; break;
			}
		}

		NeuronLayer::NeuronLayer(const std::string& Name, int nInput, int nOutput, LayerActivation Activate, TrainScheme Train, float WeightMax):
			NeuronLayer(nInput, nOutput, Activate, Train, WeightMax) {
			name = Name;
		}

		NeuronLayer::NeuronLayer(NeuralLayer_Base * inp, int nOutput, LayerActivation Activate, TrainScheme Train, float WeightMax) :
			NeuronLayer(inp->nOutput(), nOutput, Activate, Train, WeightMax) {
			PreviousLayer = inp;
		}

		NeuronLayer::NeuronLayer(const std::string& Name, NeuralLayer_Base* inp, int nOutput, LayerActivation Activate, TrainScheme Train, float WeightMax):
			NeuronLayer(inp, nOutput, Activate, Train, WeightMax) {
			name = Name;
		}

		void NeuralLayer_Base::UseDropout(int b) {
			PVX::DeepNeuralNets::UseDropout = b;
		}
		void NeuronLayer::FeedForward(int Version) {
			if (Version > FeedVersion) {
				PreviousLayer->FeedForward(Version);
				auto inp = PreviousLayer->Output();
				if (inp.cols() != output.cols()) {
					output = Eigen::MatrixXf::Ones(output.rows(), inp.cols());
				}
				if (PVX::DeepNeuralNets::UseDropout && _Dropout < 1.0f) {
					outPart(output) = 
						Activate(Weights * inp).array() * 
						(RandomBias(output.rows() - 1ll, output.cols()) < _Dropout).cast<float>() * 
						_iDropout;
				} else {
					outPart(output) = Activate(Weights * inp);
				}
				FeedVersion = Version;
			}
		}

		void NeuronLayer::BackPropagate(const Eigen::MatrixXf & Gradient) {
			Eigen::MatrixXf grad = Gradient.array() * Derivative(outPart(output)).array();
			Eigen::MatrixXf prop = Weights.transpose() * grad;
			PreviousLayer->BackPropagate(outPart(prop));
			(this->*updateWeights)(grad);
		}

		size_t NeuronLayer::nInput() {
			return Weights.cols();
		}

	}
}