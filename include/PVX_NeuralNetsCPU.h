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

			netData output;
			int FeedVersion = -1;
			static float
				__LearnRate,
				__Momentum,
				__iMomentum,
				__RMSprop,
				__iRMSprop,
				__Dropout,
				__iDropout,
				__L2;

			void Gather(std::set<NeuralLayer_Base*>& g);

			friend class NeuralNetOutput_Base;
			friend class OutputLayer;
			friend class NeuralNetContainer;
			
			virtual void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) = 0;

			void FixInputs(const std::vector<NeuralLayer_Base*>& ids);
			std::string name;
		public:
			const std::string& Name() { return name; };
			virtual void DNA(std::map<void*, WeightData> & Weights) = 0;
			void Input(NeuralLayer_Base*);
			void Inputs(const std::vector<NeuralLayer_Base*>&);
			virtual void FeedForward(int) = 0;
			virtual void BackPropagate(const netData &) = 0;
			virtual size_t nInput() = 0;

			int nOutput();
			int BatchSize();
			netData Output();
			netData RealOutput();

			static float LearnRate();
			static void LearnRate(float Alpha);
			static float Momentum();
			static void Momentum(float Beta);
			static float RMSprop();
			static void RMSprop(float Beta);
			static void L2Regularization(float lambda);
			static float Dropout();
			static void Dropout(float Rate);
			static void UseDropout(int);
			
			virtual void SetLearnRate(float a) = 0;
			virtual void ResetMomentum() = 0;
		};

		netData Concat(const std::vector<netData>& m);
		std::vector<float> Divercity(netData& a);
		netData DivercitySort(netData& a, const std::vector<float> & div);


		class InputLayer : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			InputLayer(PVX::BinLoader& bin);
		public:
			void DNA(std::map<void*, WeightData> & Weights) {};
			InputLayer(const size_t Size);
			InputLayer(const std::string& Name, const size_t Size);
			int Input(const float * Data, int Count = 1);
			int Input(const netData & Data);
			void FeedForward(int) {}
			void BackPropagate(const netData & Gradient) {}
			size_t nInput();

			void InputRaw(const netData & Data);
			netData MakeRawInput(const netData & Data);
			netData MakeRawInput(const float* Data, int Count = 1);
			netData MakeRawInput(const std::vector<float>& Input);

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
			friend class NeuralNetContainer;
			netData Weights;
			netData DeltaWeights;
			netData RMSprop;
			netData(*Activate)(const netData & Gradient);
			netData(*Derivative)(const netData & Gradient);

			void AdamF(const netData & Gradient);
			void MomentumF(const netData & Gradient);
			void RMSpropF(const netData & Gradient);
			void SgdF(const netData & Gradient);
			void AdaGradF(const netData & Gradient);

			void Adam_WeightDecayF(const netData& Gradient);
			void Momentum_WeightDecayF(const netData& Gradient);
			void RMSprop_WeightDecayF(const netData& Gradient);
			void Sgd_WeightDecayF(const netData& Gradient);
			void AdaGrad_WeightDecayF(const netData& Gradient);

			void(NeuronLayer::*updateWeights)(const netData & Gradient);
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
				_iDropout,
				_L2;
		public:
			NeuronLayer(int nInput, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			NeuronLayer(const std::string& Name, int nInput, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			NeuronLayer(NeuralLayer_Base * inp, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			NeuronLayer(const std::string& Name, NeuralLayer_Base * inp, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			size_t nInput();

			void FeedForward(int Version);
			void BackPropagate(const netData & TrainData);

			void DNA(std::map<void*, WeightData> & Weights);
			void SetLearnRate(float a);
			void ResetMomentum();

			netData & GetWeights();
		};

		class ActivationLayer :NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			netData(*Activate)(const netData& Gradient);
			netData(*Derivative)(const netData& Gradient);
			LayerActivation activation;

			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			static ActivationLayer* Load2(PVX::BinLoader & bin);
		public:
			ActivationLayer(NeuralLayer_Base* inp, LayerActivation Activation = LayerActivation::ReLU);
			ActivationLayer(int inp, LayerActivation Activation = LayerActivation::ReLU);
			
			void FeedForward(int Version);
			void BackPropagate(const netData& TrainData);
			void DNA(std::map<void*, WeightData>& Weights);
			void SetLearnRate(float a);
			void ResetMomentum();

			size_t nInput();
		};

		class NeuronAdder : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
		public:
			NeuronAdder(const int InputSize);
			NeuronAdder(const std::vector<NeuralLayer_Base*> & Inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const netData & Gradient);
			size_t nInput();

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NeuronMultiplier : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
		public:
			NeuronMultiplier(const int inputs);
			NeuronMultiplier(const std::vector<NeuralLayer_Base*> & inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const netData & Gradient);
			size_t nInput();

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NeuronCombiner : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
		public:
			NeuronCombiner(const int inputs);
			NeuronCombiner(const std::vector<NeuralLayer_Base*> & inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const netData & Gradient);
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
			netData output;
			float Error = -1.0f;
			int Version = 0;
			NetDNA Checkpoint;
			float CheckpointError = -1.0f;
			std::vector<float> CheckpointDNA;
			std::set<NeuralLayer_Base*> Gather();

			virtual void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) = 0;

			virtual void FeedForward();
			
			friend class NeuralNetContainer;
		public:
			NeuralNetOutput_Base(NeuralLayer_Base * Last);
			virtual float GetError(const netData & Data) = 0;
			virtual float Train(const float * Data) = 0;
			virtual float Train(const netData & Data) = 0;
			void Result(float * Result);
			const netData& Result();
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
			netData output;
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
			float GetError_MeanSquare(const netData & Data);
			float Train_MeanSquare(const netData & Data);
			float GetError_SoftMax(const netData & Data);
			float Train_SoftMax(const netData & Data);

			std::function<void()> FeedForward;
			std::function<float(const netData&)> GetErrorFnc;
			std::function<float(const netData&)> TrainFnc;
		public:
			OutputLayer(NeuralLayer_Base * Last, OutputType Type = OutputType::MeanSquare);
			float GetError(const netData & Data);
			float Train(const netData & Data);
			float Train(const float * Data);
			void Result(float * Result);
			const netData& Result();
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
			std::vector<netData> InputData;
			netData TrainData;
			std::vector<int> TrainOrder;
		public:
			NeuralNetContainer(OutputLayer* OutLayer);
			NeuralNetContainer(const std::wstring& Filename);
			~NeuralNetContainer();
			void Save(const std::wstring& Filename);
			void SaveCheckpoint();
			float LoadCheckpoint();
			void ResetMomentum();

			netData MakeRawInput(const netData& inp);
			netData MakeRawInput(const std::vector<float>& inp);
			std::vector<netData> MakeRawInput(const std::vector<netData>& inp);
			netData FromVector(const std::vector<float>& Data);

			std::vector<float> ProcessVec(const std::vector<float>& Inp);

			netData Process(const netData& inp);
			netData Process(const std::vector<netData>& inp);
			netData ProcessRaw(const netData& inp);
			netData ProcessRaw(const std::vector<netData>& inp);

			float Train(const netData& inp, const netData& outp);
			float TrainRaw(const netData& inp, const netData& outp);
			float Train(const std::vector<netData>& inp, const netData& outp);
			float TrainRaw(const std::vector<netData>& inp, const netData& outp);

			float Error(const netData& inp, const netData& outp);
			float ErrorRaw(const netData& inp, const netData& outp);
			float Error(const std::vector<netData>& inp, const netData& outp);
			float ErrorRaw(const std::vector<netData>& inp, const netData& outp);

			std::vector<std::pair<float*, size_t>> MakeDNA();

			void AddTrainData(const netData& inp, const netData& outp);
			void AddTrainData(const std::vector<netData>& inp, const netData& outp);
			float Iterate();
		};
	}
}

#endif