#ifndef __PVX_DEEPNEURALNETS_H__
#define __PVX_DEEPNEURALNETS_H__

#include <Eigen/dense>
#include <vector>
#include <PVX_BinSaver.h>
#include <string>
#include <random>
#include <set>
#include <map>

namespace PVX {
	namespace DeepNeuralNets {
		using netData = Eigen::MatrixXf;
		enum class LayerActivation {
			Tanh,
			TanhBias,
			ReLU,
			Sigmoid,
			Linear
		};

		struct WeightData {
			float * Weights;
			int Offset;
			int Count;
		};

		class NeuralLayer_Base {
		protected:
			NeuralLayer_Base* PreviousLayer = nullptr;
			std::vector<NeuralLayer_Base*> InputLayers;

			Eigen::MatrixXf output;
			int FeedVersion = -1;
			static float
				__LearnRate,
				__Momentum,
				__iMomentum,
				__RMSprop,
				__iRMSprop,
				__Dropout,
				__iDropout;

			void Gather(std::set<NeuralLayer_Base*>& g);

			friend class NeuralNetOutput_Base;
			friend class OutputLayer;
			friend class NeuralNetContainer;
			friend class NeuralNetContainer_Old;
			virtual void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) = 0;

			void FixInputs(const std::vector<NeuralLayer_Base*>& ids);
			std::string name;
		public:
			const std::string& Name() { return name; };
			virtual void DNA(std::map<void*, WeightData> & Weights) = 0;
			void Input(NeuralLayer_Base*);
			void Inputs(const std::vector<NeuralLayer_Base*>&);
			virtual void FeedForward(int) = 0;
			virtual void BackPropagate(const Eigen::MatrixXf &) = 0;
			virtual size_t nInput() = 0;

			int nOutput();
			int BatchSize();
			Eigen::MatrixXf Output();
			Eigen::MatrixXf RealOutput();

			static float LearnRate();
			static void LearnRate(float Alpha);
			static float Momentum();
			static void Momentum(float Beta);
			static float RMSprop();
			static void RMSprop(float Beta);
			static float Dropout();
			static void Dropout(float Rate);
			static void UseDropout(int);

			virtual void SetLearnRate(float a) = 0;
			virtual void ResetMomentum() = 0;
		};

		Eigen::MatrixXf Concat(const std::vector<Eigen::MatrixXf>& m);
		std::vector<float> Divercity(Eigen::MatrixXf& a);
		Eigen::MatrixXf DivercitySort(Eigen::MatrixXf& a, const std::vector<float> & div);


		class InputLayer : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer_Old;
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			InputLayer(PVX::BinLoader& bin);
		public:
			void DNA(std::map<void*, WeightData> & Weights) {};
			InputLayer(const size_t Size);
			InputLayer(const std::string& Name, const size_t Size);
			int Input(const float * Data, int Count = 1);
			int Input(const Eigen::MatrixXf & Data);
			void FeedForward(int) {}
			void BackPropagate(const Eigen::MatrixXf & Gradient) {}
			size_t nInput();

			void InputRaw(const Eigen::MatrixXf & Data);
			Eigen::MatrixXf MakeRawInput(const Eigen::MatrixXf & Data);
			Eigen::MatrixXf MakeRawInput(const float* Data, int Count = 1);
			Eigen::MatrixXf MakeRawInput(const std::vector<float>& Input);

			void SetLearnRate(float a) {};
			void ResetMomentum() {};
		};

		enum class TrainScheme {
			Adam,
			RMSprop,
			Momentum,
			AdaGrad,
			Sgd
		};
		class NeuronLayer : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer_Old;
			friend class NeuralNetContainer;
			Eigen::MatrixXf Weights;
			Eigen::MatrixXf DeltaWeights;
			Eigen::MatrixXf RMSprop;
			Eigen::MatrixXf(*Activate)(const Eigen::MatrixXf & Gradient);
			Eigen::MatrixXf(*Derivative)(const Eigen::MatrixXf & Gradient);

			void AdamF(const Eigen::MatrixXf & Gradient);
			void MomentumF(const Eigen::MatrixXf & Gradient);

			void RMSpropF(const Eigen::MatrixXf & Gradient);
			void SgdF(const Eigen::MatrixXf & Gradient);
			void AdaGradF(const Eigen::MatrixXf & Gradient);

			void(NeuronLayer::*updateWeights)(const Eigen::MatrixXf & Gradient);
			TrainScheme training;
			LayerActivation activation;

			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			static NeuralLayer_Base* Load2(PVX::BinLoader& bin);

			float
				_LearnRate,
				_Momentum,
				_iMomentum,
				_RMSprop,
				_iRMSprop,
				_Dropout,
				_iDropout;
		public:
			NeuronLayer(int nInput, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			NeuronLayer(const std::string& Name, int nInput, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			NeuronLayer(NeuralLayer_Base * inp, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			NeuronLayer(const std::string& Name, NeuralLayer_Base * inp, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			size_t nInput();

			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf & TrainData);

			void DNA(std::map<void*, WeightData> & Weights);
			void SetLearnRate(float a);
			void ResetMomentum();

			Eigen::MatrixXf & GetWeights();
		};

		class ActivationLayer :NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer_Old;
			friend class NeuralNetContainer;
			Eigen::MatrixXf(*Activate)(const Eigen::MatrixXf& Gradient);
			Eigen::MatrixXf(*Derivative)(const Eigen::MatrixXf& Gradient);
			LayerActivation activation;

			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			static ActivationLayer* Load2(PVX::BinLoader & bin);
		public:
			ActivationLayer(NeuralLayer_Base* inp, LayerActivation Activation = LayerActivation::ReLU);
			ActivationLayer(int inp, LayerActivation Activation = LayerActivation::ReLU);

			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf& TrainData);
			void DNA(std::map<void*, WeightData>& Weights);
			void SetLearnRate(float a);
			void ResetMomentum();

			size_t nInput();
		};

		class NeuronAdder : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer_Old;
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
		public:
			NeuronAdder(const int InputSize);
			NeuronAdder(const std::vector<NeuralLayer_Base*> & Inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf & Gradient);
			size_t nInput();

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NeuronMultiplier : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer_Old;
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
		public:
			NeuronMultiplier(const int inputs);
			NeuronMultiplier(const std::vector<NeuralLayer_Base*> & inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf & Gradient);
			size_t nInput();

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NeuronCombiner : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer_Old;
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
		public:
			NeuronCombiner(const int inputs);
			NeuronCombiner(const std::vector<NeuralLayer_Base*> & inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf & Gradient);
			size_t nInput();

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NetDNA {
			std::vector<WeightData> Layers;
			int Size = 0;
			friend class NeuralNetOutput_Base;
			friend class OutputLayer;
		public:
			std::vector<float> GetData();
			void SetData(const float * Data);
		};

		class NeuralNetOutput_Base {
		protected:
			NeuralLayer_Base * LastLayer;
			Eigen::MatrixXf output;
			float Error = -1.0f;
			int Version = 0;
			NetDNA Checkpoint;
			float CheckpointError = -1.0f;
			std::vector<float> CheckpointDNA;
			std::set<NeuralLayer_Base*> Gather();

			virtual void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) = 0;

			virtual void FeedForward();
			friend class NeuralNetContainer_Old;
			friend class NeuralNetContainer;
		public:
			NeuralNetOutput_Base(NeuralLayer_Base * Last);
			virtual float GetError(const Eigen::MatrixXf & Data) = 0;
			virtual float Train(const float * Data) = 0;
			virtual float Train(const Eigen::MatrixXf & Data) = 0;
			void Result(float * Result);
			const Eigen::MatrixXf& Result();
			int nOutput();

			void SaveCheckpoint();
			float LoadCheckpoint();

			void ResetMomentum();

			NetDNA GetDNA();
		};

		enum class OutputType {
			MeanSquare,
			SoftMax,
			StableSoftMax
		};
		class OutputLayer {
		protected:
			NeuralLayer_Base * LastLayer;
			Eigen::MatrixXf output;
			float Error = -1.0f;
			int Version = 0;
			NetDNA Checkpoint;
			float CheckpointError = -1.0f;
			std::vector<float> CheckpointDNA;
			std::set<NeuralLayer_Base*> Gather();

			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);

			friend class NeuralNetContainer_Old;
			friend class NeuralNetContainer;
			OutputType Type;

			void FeedForwardMeanSquare();
			void FeedForwardSoftMax();
			void FeedForwardStableSoftMax();
			float GetError_MeanSquare(const Eigen::MatrixXf & Data);
			float Train_MeanSquare(const Eigen::MatrixXf & Data);
			float GetError_SoftMax(const Eigen::MatrixXf & Data);
			float Train_SoftMax(const Eigen::MatrixXf & Data);

			std::function<void()> FeedForward;
			std::function<float(const Eigen::MatrixXf&)> GetErrorFnc;
			std::function<float(const Eigen::MatrixXf&)> TrainFnc;
		public:
			OutputLayer(NeuralLayer_Base * Last, OutputType Type = OutputType::MeanSquare);
			float GetError(const Eigen::MatrixXf & Data);
			float Train(const Eigen::MatrixXf & Data);
			float Train(const float * Data);
			void Result(float * Result);
			const Eigen::MatrixXf& Result();
			int nOutput();

			void SaveCheckpoint();
			float LoadCheckpoint();

			void ResetMomentum();

			NetDNA GetDNA();
		};
		class NeuralNetContainer {
		protected:
			std::vector<NeuralLayer_Base*> Layers;
			std::vector<InputLayer*> Inputs;
			OutputLayer* Output = nullptr;
			std::vector<std::pair<float*, size_t>> MakeDNA();
		public:
			NeuralNetContainer(OutputLayer* OutLayer);
			NeuralNetContainer(const std::wstring& Filename);
			~NeuralNetContainer();
			void Save(const std::wstring& Filename);
			void SaveCheckpoint();
			float LoadCheckpoint();
			void ResetMomentum();

			Eigen::MatrixXf MakeRawInput(const Eigen::MatrixXf& inp);
			Eigen::MatrixXf MakeRawInput(const std::vector<float>& inp);
			std::vector<Eigen::MatrixXf> MakeRawInput(const std::vector<Eigen::MatrixXf>& inp);
			Eigen::MatrixXf FromVector(const std::vector<float>& Data);

			std::vector<float> ProcessVec(const std::vector<float>& Inp);

			Eigen::MatrixXf Process(const Eigen::MatrixXf& inp);
			Eigen::MatrixXf Process(const std::vector<Eigen::MatrixXf>& inp);
			Eigen::MatrixXf ProcessRaw(const Eigen::MatrixXf& inp);
			Eigen::MatrixXf ProcessRaw(const std::vector<Eigen::MatrixXf>& inp);

			float Train(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float TrainRaw(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float Train(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);
			float TrainRaw(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);

			float Error(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float ErrorRaw(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float Error(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);
			float ErrorRaw(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);
		};
		class MeanSquareOutput : public NeuralNetOutput_Base {
		protected:
			friend class NeuralNetContainer_Old;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			MeanSquareOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*> & Prevs);
		public:
			MeanSquareOutput(NeuralLayer_Base * Last);
			float GetError(const Eigen::MatrixXf & Data);
			float Train(const float * Data);
			float Train(const Eigen::MatrixXf & Data);
			float Train2(const Eigen::MatrixXf & Data);
		};

		class SoftmaxOutput : public NeuralNetOutput_Base {
		protected:
			friend class NeuralNetContainer_Old;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			void FeedForward();
			SoftmaxOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*>& Prevs);
		public:
			SoftmaxOutput(NeuralLayer_Base * Last);
			float GetError(const Eigen::MatrixXf & Data);
			float Train(const float * Data);
			float Train(const Eigen::MatrixXf & Data);
		};

		class StableSoftmaxOutput : public NeuralNetOutput_Base {
		protected:
			friend class NeuralNetContainer_Old;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			void FeedForward();
			StableSoftmaxOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*>& Prevs);
		public:
			StableSoftmaxOutput(NeuralLayer_Base * Last);
			float GetError(const Eigen::MatrixXf & Data);
			float Train(const float * Data);
			float Train(const Eigen::MatrixXf & Data);
		};

		class NeuralNetContainer_Old {
		protected:
			std::vector<NeuralLayer_Base*> Layers;
			std::vector<InputLayer*> Inputs;
			NeuralNetOutput_Base* Output = nullptr;
			std::vector<std::pair<float*, size_t>> MakeDNA();
		public:
			NeuralNetContainer_Old(NeuralNetOutput_Base* OutLayer);
			NeuralNetContainer_Old(const std::wstring& Filename);
			~NeuralNetContainer_Old();
			void Save(const std::wstring& Filename);
			void SaveCheckpoint();
			float LoadCheckpoint();
			void ResetMomentum();

			Eigen::MatrixXf MakeRawInput(const Eigen::MatrixXf& inp);
			Eigen::MatrixXf MakeRawInput(const std::vector<float>& inp);
			std::vector<Eigen::MatrixXf> MakeRawInput(const std::vector<Eigen::MatrixXf>& inp);
			Eigen::MatrixXf FromVector(const std::vector<float>& Data);

			std::vector<float> ProcessVec(const std::vector<float>& Inp);

			Eigen::MatrixXf Process(const Eigen::MatrixXf& inp);
			Eigen::MatrixXf Process(const std::vector<Eigen::MatrixXf>& inp);
			Eigen::MatrixXf ProcessRaw(const Eigen::MatrixXf& inp);
			Eigen::MatrixXf ProcessRaw(const std::vector<Eigen::MatrixXf>& inp);

			float Train(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float TrainRaw(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float Train(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);
			float TrainRaw(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);

			float Error(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float ErrorRaw(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float Error(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);
			float ErrorRaw(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);
		};
	}
}

#endif