#include <PVX_NeuralNetsCPU.h>
#include <iostream>
#include <future>

namespace PVX {
	namespace DeepNeuralNets {
		void NeuronAdder::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const {
			bin.Begin("ADDR");
			{
				bin.Write("INPC", int(nInput()));
				bin.Begin("LYRS"); {
					for (auto& i: InputLayers)
						bin.write(int(IndexOf.at(i)));
				} bin.End();
			}
			bin.End();
		}
		NeuronAdder* NeuronAdder::Load2(PVX::BinLoader& bin) {
			int inp;
			std::vector<int> layers;
			bin.Read("INPC", inp);
			bin.Read("LYRS", layers);
			bin.Execute();
			auto add = new NeuronAdder(inp);
			for (auto l : layers) {
				add->InputLayers.push_back(reinterpret_cast<NeuralLayer_Base*>(l));
			}
			return add;
		}
		NeuronAdder::NeuronAdder(const int InputSize) {
			output = netData::Zero(InputSize + 1, 1);
		}
		NeuronAdder::NeuronAdder(const std::vector<NeuralLayer_Base*>& Inputs) : NeuronAdder(Inputs[0]->nOutput()) {
			InputLayers = Inputs;
		}
		void NeuronAdder::DNA(std::map<void*, WeightData>& Weights) {
			for (auto l : InputLayers)
				l->DNA(Weights);
		}
		void NeuronAdder::FeedForward(int Version) {
			if (Version > FeedVersion) {
				InputLayers[0]->FeedForward(Version);
				output = InputLayers[0]->Output();
				for (auto i = 1; i < InputLayers.size(); i++) {
					InputLayers[i]->FeedForward(Version);
					output += InputLayers[i]->Output();
				}
				output.row(output.rows() - 1) = netData::Ones(1, output.cols());
				FeedVersion = Version;
			}
		}
		void NeuronAdder::BackPropagate(const netData & Gradient) {
			for (auto i : InputLayers) i->BackPropagate(Gradient);
		}
		size_t NeuronAdder::nInput() const {
			return output.rows() - 1;
		}

		void NeuronAdder::SetLearnRate(float a) {
			for (auto l : InputLayers)
				l->SetLearnRate(a);
		}

		void NeuronAdder::ResetMomentum() {
			for (auto i : InputLayers)
				i->ResetMomentum();
		}

		void NeuronCombiner::SetLearnRate(float a) {
			for (auto l : InputLayers)
				l->SetLearnRate(a);
		}

		void NeuronCombiner::ResetMomentum() {
			for (auto i : InputLayers)
				i->ResetMomentum();
		}

		void NeuronMultiplier::Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const {
			bin.Begin("MULP");
			{
				for (auto& i: InputLayers)
					bin.write(int(IndexOf.at(i)));
			}
			bin.End();
		}

		NeuronMultiplier::NeuronMultiplier(const int inputs) {
			output = netData::Zero(inputs + 1, 1);
		}

		NeuronMultiplier::NeuronMultiplier(const std::vector<NeuralLayer_Base*> & inputs) {
			for (auto& i : inputs) InputLayers.push_back(i);
			output = netData::Zero(InputLayers[0]->Output().rows(), 1);
		}
		void NeuronMultiplier::DNA(std::map<void*, WeightData>& Weights) {
			for (auto l : InputLayers)
				l->DNA(Weights);
		}
		void NeuronMultiplier::FeedForward(int Version) {
			if (Version > FeedVersion) {
				InputLayers[0]->FeedForward(Version);
				auto tmp = InputLayers[0]->Output().array();
				for (auto i = 1; i < InputLayers.size(); i++) {
					InputLayers[i]->FeedForward(Version);
					tmp *= InputLayers[i]->Output().array();
				}
				output = tmp;
				FeedVersion = Version;
			}
		}
		void NeuronMultiplier::BackPropagate(const netData & Gradient) {
			{
				auto tmp = InputLayers[1]->RealOutput().array();
				for (auto i = 2; i < InputLayers.size(); i++) {
					tmp *= InputLayers[i]->RealOutput().array();
				}
				InputLayers[0]->BackPropagate(Gradient.array() * tmp);
			}
			for (int j = 1; j < InputLayers.size(); j++) {
				auto tmp = InputLayers[0]->RealOutput().array();
				for (int i = 1; i < InputLayers.size(); i++)
					if (i != j) 
						tmp *= InputLayers[i]->RealOutput().array();
				InputLayers[j]->BackPropagate(Gradient.array() * tmp);
			}
		}
		size_t NeuronMultiplier::nInput() const {
			return output.cols();
		}

		void NeuronMultiplier::SetLearnRate(float a) {
			for (auto i : InputLayers)
				i->SetLearnRate(a);
		}
		void NeuronMultiplier::ResetMomentum() {
			for (auto i : InputLayers)
				i->ResetMomentum();
		}

		netData Concat(const std::vector<netData>& M) {
			int cols = 0;
			for (auto& m : M) cols += m.cols();
			netData ret(M[0].rows(), cols);
			size_t offset = 0;
			for (auto& m : M) {
				memcpy(ret.data() + offset, m.data(), m.size() * sizeof(float));
				offset += m.size();
			}
			return ret;
		}

		std::vector<float> Diverse(netData& a) {
			netData norm(a.rows(), a.cols());
			for (auto i = 0; i<a.cols(); i++)
				norm.col(i) = a.col(i).normalized();

			std::vector<std::pair<float, Eigen::RowVectorXf>> n;
			n.reserve(norm.cols());
			for (auto i = 0; i<a.cols(); i++) {
				float max = 0.0f;
				for (auto j = 0; j<i; j++) {
					float dot = (norm.col(i).transpose() * norm.col(j))(0, 0);
					if (dot > max) max = dot;
				}
				n.emplace_back(max, a.col(i));
			}
			std::sort(n.begin(), n.end(), [](auto a, auto b) { return a.first < b.first; });

			std::vector<float> ret;
			ret.reserve(norm.cols());
			int cc = 0;
			for (auto & [d, c]: n) {
				norm.col(cc++) = c;
				ret.push_back(d);
			}
			a = norm;
			return ret;
		}

		std::vector<float> Divercity(netData& a) {
			netData norm(a.rows(), a.cols());
			for (auto i = 0; i<a.cols(); i++)
				norm.col(i) = a.col(i).normalized();

			std::vector<float> n;
			n.reserve(norm.cols());
			for (auto i = 0; i<a.cols(); i++) {
				float max = 0.0f;
				for (auto j = 0; j<i; j++) {
					float dot = (norm.col(i).transpose() * norm.col(j))(0, 0);
					if (dot > max) max = dot;
				}
				n.push_back(max);
			}
			return n;
		}
		netData DivercitySort(netData& a, const std::vector<float>& div) {
			std::vector<std::pair<float, Eigen::RowVectorXf>> all;
			for (auto i = 0; i< div.size(); i++)
				all.emplace_back(div[i], a.col(i));
			
			std::sort(all.begin(), all.end(), [](auto a, auto b) { return a.first < b.first; });
			netData ret(a.rows(), a.cols());
			for (auto i = 0; i< all.size(); i++)
				ret.col(i) = all[i].second;
			return ret;
		}
	}
}